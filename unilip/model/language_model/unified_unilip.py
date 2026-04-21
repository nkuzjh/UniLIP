from abc import ABC, abstractmethod
import math
import copy
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import cv2
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

# Transformers & Diffusers
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, InternVLConfig, InternVLModel, InternVLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils.torch_utils import randn_tensor

# Local Imports (Assuming these exist in your project structure)
from ..sana import build_sana
from ..vae_modules import DCAE_Decoder
from unilip.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_IDX, DEFAULT_IM_START_TOKEN_IDX, DEFAULT_IM_END_TOKEN_IDX, UND_IMAGE_TOKEN_IDX

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel



import torch
import transformers.modeling_utils


from unilip.model.language_model.autograd_check import (
    inspect_loss_routes_with_autograd_grad,
    print_loss_route_summary,
    print_optimizer_managed_param_summary,
)

# =================================================================
# [Monkey Patch] 强制所有 Gradient Checkpointing 使用 use_reentrant=False
# 修复 "Trying to backward through the graph a second time" 错误
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



# [NEW] Helper Functions from Pi0.5 (Ported)
def get_safe_dtype(target_dtype, device_type):
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))
# 逐行解释
#   alpha_t = torch.as_tensor(alpha, ...)将 alpha 参数（通常是一个 Python 浮点数，例如 0.5）转换为一个 PyTorch 张量。
#   beta_t = torch.as_tensor(beta, ...)将 beta 参数转换为一个 PyTorch 张量。
#   dist = torch.distributions.Beta(alpha_t, beta_t)这是核心代码。它创建了一个 Beta 分布的实例 (object)。这个实例 "知道" 它的形状是由 alpha 和 beta 定义的。
#   return dist.sample((bsize,))调用该分布实例的 .sample() 方法，从中抽取 bsize 个随机样本。返回的张量形状为 [bsize]，其中的每个值都是 0.0 到 1.0 之间的一个浮点数。
# $\alpha$ 和 $\beta$ 的作用
    # alpha 和 beta 控制着分布的“偏好”：
    #   alpha = 1, beta = 1:这是均匀分布 (Uniform Distribution)。dist.sample() 的行为与 torch.rand(bsize) 相同（0 到 1 之间的完全随机数）。
    #   alpha > 1, beta > 1:分布呈钟形，样本会聚集在中间（例如，alpha=5, beta=5 时，样本会聚集在 0.5 附近）。
    #   alpha < 1, beta < 1:分布呈 U 形（浴缸形状），样本会聚集在两端（0.0 或 1.0 附近）。
    #   alpha > beta (例如 alpha=5, beta=1):分布向右偏移，样本会聚集在 1.0 附近。
    #   alpha < beta (例如 alpha=1, beta=5):分布向左偏移，样本会聚集在 0.0 附近。
# 常见用途: 这个函数经常用于像 MixUp 这样的数据增强技术中，用来生成一个非均匀的混合比例（$\lambda$）。

# 手动初始化权重 (使用 Xavier 或 Small Normal)
def init_weights(m):
    if isinstance(m, nn.Linear):
        # 使用较小的标准差，确保初始输出接近 0
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Aution_AE.decoder初始化0.002
def small_init_weights(m):
    if isinstance(m, nn.Linear):
        # 使用较小的标准差，确保初始输出接近 0
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class REPAMLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 3, hidden_ratio: float = 1.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"REPA projector num_layers must be >= 1, got {num_layers}")

        hidden_dim = max(int(in_dim * hidden_ratio), 1)
        layers = []
        prev_dim = in_dim
        for layer_idx in range(num_layers - 1):
            next_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.SiLU())
            prev_dim = next_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REPASpatialNorm(nn.Module):
    def __init__(self, gamma: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        mean = x_float.mean(dim=(2, 3), keepdim=True)
        std = x_float.std(dim=(2, 3), keepdim=True, unbiased=False)
        x_float = x_float - self.gamma * mean
        x_float = x_float / (std + self.eps)
        return x_float.to(dtype=x.dtype)


class REPAConvSpatialNormProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        expected_num_patches: int,
        kernel_size: int = 3,
        use_spatial_norm: bool = True,
        spatial_norm_gamma: float = 1.0,
    ):
        super().__init__()
        if kernel_size != 3:
            raise ValueError(f"Current REPA conv projector only supports kernel_size=3, got {kernel_size}")

        grid_size = math.isqrt(expected_num_patches)
        if grid_size * grid_size != expected_num_patches:
            raise ValueError(
                f"REPA conv projector requires square patch layout, got expected_num_patches={expected_num_patches}"
            )

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.expected_num_patches = int(expected_num_patches)
        self.grid_size = int(grid_size)
        self.use_spatial_norm = bool(use_spatial_norm)
        self.spatial_norm = REPASpatialNorm(gamma=spatial_norm_gamma) if self.use_spatial_norm else None
        self.proj = nn.Conv2d(self.in_dim, self.out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        nn.init.normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"REPA conv projector expects [B, T, C], got shape={tuple(x.shape)}")

        batch_size, num_tokens, hidden_size = x.shape
        if num_tokens != self.expected_num_patches:
            raise RuntimeError(
                f"Unexpected REPA student token count in conv projector: got {num_tokens}, expected {self.expected_num_patches}."
            )
        if hidden_size != self.in_dim:
            raise RuntimeError(
                f"Unexpected REPA student hidden size in conv projector: got {hidden_size}, expected {self.in_dim}."
            )

        x = x.transpose(1, 2).contiguous().view(batch_size, hidden_size, self.grid_size, self.grid_size)
        if self.spatial_norm is not None:
            x = self.spatial_norm(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x



class AttentionPooler(nn.Module):
    """
    第一步：用于将 (bs, N, dim) 的 FPS 特征压缩为 (bs, 1, dim) 的 Query Token
    """
    def __init__(self, dim):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x: [bs, seq_len, dim]
        scores = self.attn_mlp(x)                  # [bs, seq_len, 1]
        weights = F.softmax(scores, dim=1)         # [bs, seq_len, 1]
        pooled = torch.sum(x * weights, dim=1)     # [bs, dim]
        return pooled.unsqueeze(1)                 # [bs, 1, dim]

class CrossViewFusionModule(nn.Module):
    def __init__(self, dim=1024, num_heads=8, dropout=0.1):
        """
        特征融合模块
        :param dim: 特征维度 (如 InternViT 的 1024)
        :param num_heads: 多头注意力的头数
        :param dropout: Dropout 概率
        """
        super().__init__()

        # 1. FPS 降维池化层 (负责生成具有全局视野的 Query)
        self.fps_pooler = AttentionPooler(dim)

        # 2. 跨模态注意力层 (Cross-Attention)
        # 注意：batch_first=True 允许输入形状为 (batch, seq, feature)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. 规范化层 (Pre-LN 架构，训练更稳定)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        # 4. 前馈神经网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, fps_feat, map_feat):
        """
        :param fps_feat: [bs, patch_num*patch_num, dim] (例如 [bs, 1024, 1024])
        :param map_feat: [bs, patch_num*patch_num, dim] (例如 [bs, 1024, 1024])
        :return: [bs, 1, dim] 融合后的位置特征
        """
        # --- Step 1: 将 FPS 提纯为单一的 Query ---
        # query_fps: [bs, 1, dim]
        query_fps = self.fps_pooler(fps_feat)

        # --- Step 2: 规范化 (Pre-LN) ---
        query_norm = self.norm_q(query_fps)
        map_norm = self.norm_kv(map_feat)

        # --- Step 3: Cross Attention 融合 ---
        # Q 来自 FPS (找什么), K 和 V 来自 Map (在哪里)
        # attn_out: [bs, 1, dim]
        # attn_weights: [bs, 1, map_seq_len] (可以用于可视化，看看模型关注了地图哪里)
        attn_out, attn_weights = self.cross_attn(
            query=query_norm,
            key=map_norm,
            value=map_norm
        )

        # 残差连接 1
        x = query_fps + attn_out

        # --- Step 4: FFN (增强非线性表达) ---
        # x: [bs, 1, dim]
        out = x + self.ffn(self.norm_ffn(x))

        # (可选) 如果你需要在外部做可视化分析，可以 return out, attn_weights
        return out



# Vibe Codex
class MultiViewCLSFeatureFusion(nn.Module):
    def __init__(self, dim, fused_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 6, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, und_cls, aux_cls, und_avg, aux_avg):
        fused = torch.cat(
            [
                und_cls,
                aux_cls,
                und_avg,
                aux_avg,
                torch.abs(und_cls - aux_cls),
                und_avg * aux_avg,
            ],
            dim=-1,
        )
        return self.net(fused)



from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

@dataclasses.dataclass
class GemmaActionExpertConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

from transformers.models.auto import CONFIG_MAPPING
from transformers import GemmaForCausalLM
import safetensors.torch
import os


# ==========================================
# 1. MetaModel: 定义组件 (Connectors & Heads)
# ==========================================
class Unified_UniLIP_InternVL_MetaModel:

    def __init__(self, config):
        super(Unified_UniLIP_InternVL_MetaModel, self).__init__(config)

        # --- A. Existing Logic (Vision Tower & LLM Setup) ---
        if hasattr(config, "n_query"):
            path = config.mllm_path
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16, # Assuming bf16 training
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.vision_tower = internvl_model.vision_model
            self.multi_modal_projector = internvl_model.mlp1

            # Disable dropout for vision tower
            for layer in self.vision_tower.encoder.layers:
                try:
                    layer.drop_path1.drop_prob = 0.0
                    layer.drop_path2.drop_prob = 0.0
                except:
                    continue
            self.vision_tower.eval()
            self.multi_modal_projector.eval()

            # Latent Queries for Generation Task
            if 'hidden_size' in self.config:
                hidden_size = self.config.hidden_size
            else:
                hidden_size = self.config.text_config.hidden_size

            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, hidden_size))

            # --- B. Generation Path (DiT & Connector) ---
            self.dit, self.vae, self.noise_scheduler = build_sana(config.dit_path)

            # VAE Decoder for Latent Space
            vae_config_dict = {'model': {'dc_ae_path': config.vae_path}}
            vae_config_omega = OmegaConf.create(vae_config_dict)
            llm_hidden_size = self.multi_modal_projector[-1].weight.shape[-1]
            self.vae_decoder = DCAE_Decoder(vae_config_omega, llm_hidden_size)

            # LLM Connector (Slice of InternVL for DiT)
            path = config.mllm_hf_path
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            self.llm_connector = copy.deepcopy(internvl_model.language_model)
            del self.llm_connector.layers[:-self.config.connect_layer]
            del self.llm_connector.embed_tokens
            self.projector = nn.Linear(llm_hidden_size, self.dit.config.caption_channels)

            # --- C. [NEW] Localization Path (Action Connector & Flow Matching Heads) ---
            pass
            # 加载完csgo_config.yaml后，在initializa_localization_modules中实现Loc Head的初始化逻辑

            # if getattr(self.config, "use_pi05_action_dit", False):
            #     logging.info(f"Init Pi0.5 Action DiT ...")
            #     action_expert_config = Config(
            #         width=1024,
            #         depth=18,
            #         mlp_dim=4096,
            #         num_heads=8,
            #         num_kv_heads=1,
            #         head_dim=256,
            #         # lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)} if self.is_lora else None,
            #     )
            #     action_expert_config_hf = GemmaConfig(
            #         head_dim=action_expert_config.head_dim,
            #         hidden_size=action_expert_config.width,
            #         intermediate_size=action_expert_config.mlp_dim,
            #         num_attention_heads=action_expert_config.num_heads,
            #         num_hidden_layers=action_expert_config.depth,
            #         num_key_value_heads=action_expert_config.num_kv_heads,
            #         vocab_size=257152,
            #         hidden_activation="gelu_pytorch_tanh",
            #         torch_dtype="bfloat16",
            #         use_adarms=True,
            #         adarms_cond_dim=action_expert_config.width,
            #     )
            #     self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
            #     self.gemma_expert.model.embed_tokens = None
            #     _default_pi05_action_dim = 32
            #     self.action_in_proj = nn.Linear(_default_pi05_action_dim, action_expert_config.width)
            #     self.action_out_proj = nn.Linear(action_expert_config.width, _default_pi05_action_dim)
            #     self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            #     self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
            #     logging.info(f"Init Pi0.5 Action DiT Complete !")
            # else:
            #     # if getattr(self.config, "is_loc_learnable_query", False):
            #     #     self.loc_learnable_query = nn.Parameter(torch.randn(1, 1, llm_hidden_size))
            #     # if getattr(self.config, "is_action_dit_projector", False):
            #     #     self.action_dit_projector = nn.Sequential(
            #     #         nn.Linear(llm_hidden_size, llm_hidden_size*4, bias=True),
            #     #         nn.GELU(),
            #     #         nn.Linear(llm_hidden_size*4, llm_hidden_size*2, bias=True),
            #     #         nn.GELU(),
            #     #         nn.Linear(llm_hidden_size*2, llm_hidden_size, bias=True),
            #     #     )
            #     # 输入action_dit前的feature首先进行normalize
            #     self.action_dit_norm = Qwen2RMSNorm(llm_hidden_size, eps=1e-6)

            #     # 1. Action Dit
            #     # 这里的 config.action_dit_layer 可以在 model_args 中定义，默认比如 3 或 6
            #     action_layers = getattr(config, "action_dit_layer", 3)
            #     # 同样使用 InternVL 的后几层切片 (复用 llm_connector 的思路)
            #     internvl_model2 = AutoModel.from_pretrained(
            #         path,
            #         torch_dtype=torch.bfloat16,
            #         low_cpu_mem_usage=True,
            #         trust_remote_code=True,
            #         attn_implementation="eager"
            #     )
            #     self.action_dit = copy.deepcopy(internvl_model2.language_model)#.to(torch.bfloat16)
            #     del self.action_dit.layers[:-action_layers]
            #     del self.action_dit.embed_tokens # 不需要 Embedding 层，直接吃 Hidden States

            #     # 2. Flow Matching Heads
            #     # 将 Action (5D Pose) 映射到 LLM Hidden Size
            #     self.action_dim = getattr(config, "action_dim", 5) # x, y, z, pitch, yaw
            #     # self.action_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)#.to(torch.bfloat16)
            #     self.action_in_proj = nn.Linear(self.action_dim, llm_hidden_size)#.to(torch.bfloat16)

            #     # 3. 时间步 MLP
            #     self.time_mlp_in = nn.Linear(llm_hidden_size, llm_hidden_size)#.to(torch.bfloat16)
            #     self.time_mlp_out = nn.Linear(llm_hidden_size, llm_hidden_size)#.to(torch.bfloat16)

            #     # 4. 输出投影 (Hidden -> Action Velocity)
            #     self.action_out_proj = nn.Linear(llm_hidden_size, self.action_dim)#.to(torch.bfloat16)

            #     # if self.config.is_loc_learnable_query:
            #     #     self.loc_learnable_query.apply(init_weights)
            #     # if self.config.is_action_dit_projector:
            #     #     self.action_dit_projector.apply(init_weights)
            #     if getattr(self.config, "is_aciton_dit_vae_small_init", False):
            #         self.action_in_proj.apply(small_init_weights)
            #     else:
            #         self.action_in_proj.apply(init_weights)
            #     self.time_mlp_in.apply(init_weights)
            #     self.time_mlp_out.apply(init_weights)
            #     if getattr(self.config, "is_aciton_dit_vae_small_init", False):
            #         self.action_out_proj.apply(small_init_weights)
            #     else:
            #         self.action_out_proj.apply(init_weights)


    def initialize_vision_modules(self, model_args, fsdp=None):
        # ... (Existing initialization logic for Vision, VAE, DiT) ...
        unilip_path = model_args.unilip_path
        self.unilip_path = unilip_path
        self.unilip_factor = model_args.unilip_factor
        self.fix_dit = model_args.fix_dit
        self.fix_connect = model_args.fix_connect
        logging.info(f"fix connect {self.fix_connect}")
        logging.info(f"fix dit {self.fix_dit}")
        if getattr(self, 'vae_decoder', None) is None:
            # replace hf structure with original internvl structure
            path = model_args.mllm_path
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True)
            self.vision_tower = internvl_model.vision_model
            self.multi_modal_projector = internvl_model.mlp1

            for layer in self.vision_tower.encoder.layers:
                try:
                    layer.drop_path1.drop_prob = 0.0
                    layer.drop_path2.drop_prob = 0.0
                except:
                    continue
            logging.info(f"should no drop out {self.vision_tower}")
            self.vision_tower.eval()

            # load unilip pretrain weight
            logging.info(f"load from unilip path {unilip_path}, factor {self.unilip_factor}")
            unilip_ckpt = torch.load(unilip_path)
            encoder_state = {}
            for key, value in unilip_ckpt.items():
                if 'encoder.' in key:
                    newkey = key[len('encoder.'):]
                    encoder_state[newkey] = value

            msg = self.vision_tower.load_state_dict(encoder_state)
            for p in self.vision_tower.parameters():
                p.requires_grad = False
            logging.info(f"load unilip vision encoder {msg}")

            mlp1_state = {}
            for key, value in unilip_ckpt.items():
                if 'mlp1.' in key:
                    newkey = key[len('mlp1.'):]
                    mlp1_state[newkey] = value
            msg = self.multi_modal_projector.load_state_dict(mlp1_state)
            for p in self.multi_modal_projector.parameters():
                p.requires_grad = False
            logging.info(f"load unilip mlp1 {msg}")

            # load vae decoder
            vae_config = {
                'model':{
                    'dc_ae_path': model_args.vae_path
                }
            }
            vae_config = OmegaConf.create(vae_config)
            llm_hidden_size = self.multi_modal_projector[-1].weight.shape[-1]
            self.vae_decoder = DCAE_Decoder(vae_config, llm_hidden_size)
            for name in list(unilip_ckpt.keys()):
                if 'regressor' in name:
                    del unilip_ckpt[name]
                else:
                    if 'decoder' in name or 'down' in name:
                        continue
                    else:
                        del unilip_ckpt[name]
            msg = self.vae_decoder.load_state_dict(unilip_ckpt)
            for p in self.vae_decoder.parameters():
                p.requires_grad = False
            logging.info(f"load unilip decoder {msg}")
        else:
            logging.info("unilip load from checkpoint!!!")
            self.vision_tower.eval()
            for p in self.vision_tower.parameters():
                p.requires_grad = False
            for p in self.multi_modal_projector.parameters():
                p.requires_grad = False
            for p in self.vae_decoder.parameters():
                p.requires_grad = False

        if getattr(self, 'dit', None) is None:
            logging.info("random initiation the DiT !!!")
            self.dit, self.vae, self.noise_scheduler = build_sana(model_args.dit_path)
        else:
            logging.info("DiT load from checkpoint!!!")
            for p in self.dit.parameters():
                p.requires_grad = True
        if self.fix_dit:
            for p in self.dit.parameters():
                p.requires_grad = False

        if getattr(self, 'llm_connector', None) is None:
            logging.info("initialize the llm connector !!!")
            path = model_args.mllm_hf_path
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=self.vision_tower.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager") # for bidr attention
            self.llm_connector = deepcopy(internvl_model.language_model)
            del self.llm_connector.layers[:-model_args.connect_layer]
            del self.llm_connector.embed_tokens
            self.projector = nn.Linear(llm_hidden_size, self.dit.config.caption_channels)
        else:
            logging.info("Connector load from checkpoint!!!")
            for p in self.llm_connector.parameters():
                p.requires_grad = True
            for p in self.projector.parameters():
                p.requires_grad = True

        self.config.n_query = model_args.n_query
        self.config.connect_layer = model_args.connect_layer
        self.config.mllm_path = model_args.mllm_path
        self.config.mllm_hf_path = model_args.mllm_hf_path
        self.config.vae_path = model_args.vae_path
        self.config.dit_path = model_args.dit_path
        self.config.unilip_factor = model_args.unilip_factor

        if getattr(self, 'latent_queries', None) is None:
            logging.info("random initiation the latent_queries !!!")
            if 'hidden_size' in self.config:
                hidden_size = self.config.hidden_size
            else:
                hidden_size = self.config.text_config.hidden_size
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, hidden_size))
        else:
            logging.info("latent_queries load from checkpoint!!!")
            self.latent_queries.requires_grad = True

        connect_require_grad = not self.fix_connect
        for p in self.llm_connector.parameters():
            p.requires_grad = connect_require_grad
        for p in self.projector.parameters():
            p.requires_grad = connect_require_grad
        self.latent_queries.requires_grad = connect_require_grad

        ### 是否开启LoRA，重新配置可学习参数
        self.is_lora = getattr(self.config, 'is_lora', False)
        # 1. Vision Tower & Multi-modal Projector
        if getattr(model_args, "train_mm_projector_only", False):
            for p in self.vision_tower.parameters(): p.requires_grad = False
            for p in self.multi_modal_projector.parameters(): p.requires_grad = True
            logging.info("train_mm_projector_only=True: freeze vision_tower and train multi_modal_projector only")
        elif not model_args.fix_vit:
            if self.is_lora:
                # LoRA模式：Vision Tower主体冻结，由PEFT接管；Projector通常全量训练(作为modules_to_save)
                for p in self.vision_tower.parameters(): p.requires_grad = False
                for p in self.multi_modal_projector.parameters(): p.requires_grad = True
            else:
                # 全量微调模式
                for p in self.vision_tower.parameters(): p.requires_grad = True
                for p in self.multi_modal_projector.parameters(): p.requires_grad = True

        # 2. LLM Backbone
        if getattr(model_args, "train_shared_llm_tail_only", False):
            for p in self.language_model.parameters():
                p.requires_grad = False

            total_layers = len(self.language_model.layers)
            tail_num_layers = min(
                max(int(getattr(model_args, "shared_llm_tail_num_layers", 2)), 0),
                total_layers,
            )
            start_layer_idx = total_layers - tail_num_layers
            for layer_idx in range(start_layer_idx, total_layers):
                for p in self.language_model.layers[layer_idx].parameters():
                    p.requires_grad = True

            logging.info(
                "train_shared_llm_tail_only=True: freeze language_model backbone except last %d layer(s) [%d:%d)",
                tail_num_layers,
                start_layer_idx,
                total_layers,
            )
        elif not model_args.fix_llm:
            if self.is_lora:
                # LoRA模式：LLM主体冻结，由PEFT接管
                for p in self.language_model.parameters(): p.requires_grad = False
            else:
                for p in self.language_model.parameters(): p.requires_grad = True

        # 3. LLM Connector (Gen Branch)
        if not self.fix_connect:
            if self.is_lora:
                # Connector是InternVL切片，视为Backbone，用LoRA训练
                for p in self.llm_connector.parameters(): p.requires_grad = False
                # Projector 是Linear映射层，建议全量训练
                for p in self.projector.parameters(): p.requires_grad = True
                self.latent_queries.requires_grad = True
            else:
                for p in self.llm_connector.parameters(): p.requires_grad = True
                for p in self.projector.parameters(): p.requires_grad = True
                self.latent_queries.requires_grad = True

        # 4. SANA DiT (Gen Branch)
        if not self.fix_dit:
            if self.is_lora:
                # DiT 也是大模型，用 LoRA
                for p in self.dit.parameters(): p.requires_grad = False
            else:
                for p in self.dit.parameters(): p.requires_grad = True


    def initialize_localization_modules(self, model_args):
        # [Simulating previous code structure for brevity]
        self.config.action_horizon = getattr(model_args, 'action_horizon', 1) # CS2 Pose is typically single step
        self.config.action_dim = getattr(model_args, 'action_dim', 5)
        self.config.is_action_dit_dense_timestep = getattr(model_args, 'is_action_dit_dense_timestep', False)
        llm_hidden_size = self.multi_modal_projector[-1].weight.shape[-1]
        # [NEW] Initialize Action Connector & Heads
        # if getattr(self, 'action_dit', None) is None:
        # 直接移植pi05的action_dit模型和权重作为定位head,无需初始化权重
        if getattr(self.config, "use_codex_vit_regression_head", False):# vibe codex
            fused_dim = self.vision_tower.config.hidden_size * 2
            self.vit_loc_fusion = MultiViewCLSFeatureFusion(
                dim=self.vision_tower.config.hidden_size,
                fused_dim=fused_dim,
                dropout=0.1,
            )
            if getattr(self.config, "is_action_dit_projector", False):
                self.action_dit_norm = Qwen2RMSNorm(fused_dim, eps=1e-6)
                self.action_dit_projector = nn.Sequential(
                    nn.Linear(fused_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                fused_dim = 256

            logging.info(f"Init ViT Codex Regression Loc Head")
            self.regression_loc_head = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.config.action_dim),
                nn.Sigmoid(),
            )
        elif getattr(self.config, "use_vit_cls_regression_head", False):
            regression_loc_head_input_dim = self.vision_tower.config.hidden_size
            if getattr(self.config, "is_action_dit_projector", False):
                self.action_dit_norm = Qwen2RMSNorm(regression_loc_head_input_dim, eps=1e-6)
                self.action_dit_projector = nn.Sequential(
                    nn.Linear(regression_loc_head_input_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                regression_loc_head_input_dim = 256

            logging.info(f"Init ViT CLS Regression Loc Head")
            self.regression_loc_head = nn.Sequential(
                nn.Linear(regression_loc_head_input_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, self.config.action_dim)
            )
        elif  getattr(self.config, "use_vit_regression_head", False):
            regression_loc_head_input_dim = llm_hidden_size
            self.cross_view_fusion = CrossViewFusionModule(dim=llm_hidden_size, num_heads=8, dropout=0.1)
            # self.img_pooler =
            if getattr(self.config, "is_action_dit_projector", False):
                # print("llm_hidden_size: ",llm_hidden_size)
                self.action_dit_norm = Qwen2RMSNorm(llm_hidden_size, eps=1e-6)
                self.action_dit_projector = nn.Sequential(
                    nn.Linear(llm_hidden_size, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                regression_loc_head_input_dim = 256

            logging.info(f"Init ViT Regression Loc Head")
            self.regression_loc_head = nn.Sequential(
                nn.Linear(regression_loc_head_input_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, self.config.action_dim)
            )

        elif getattr(self.config, "use_pi05_action_dit", False):
            if getattr(self.config, "is_action_dit_projector", False):
                self.action_dit_connector = nn.Sequential(
                    nn.Linear(llm_hidden_size, llm_hidden_size*2, bias=True),
                    nn.GELU(),
                    nn.Linear(llm_hidden_size*2, 1024*2, bias=True),
                    nn.GELU(),
                    nn.Linear(1024*2, 1024, bias=True),
                )
                self.action_dit_norm = Qwen2RMSNorm(1024, eps=1e-6)
                self.action_dit_projector = nn.Sequential(
                    nn.Linear(1024, 1024*4, bias=True),
                    nn.GELU(),
                    nn.Linear(1024*4, 1024*2, bias=True),
                    nn.GELU(),
                    nn.Linear(1024*2, 1024, bias=True),
                )

            logging.info(f"Init Pi0.5 Action DiT")
            action_expert_config = GemmaActionExpertConfig(
                width=1024,
                depth=18,
                mlp_dim=4096,
                num_heads=8,
                num_kv_heads=1,
                head_dim=256,
            )
            action_expert_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=action_expert_config.head_dim,
                hidden_size=action_expert_config.width,
                intermediate_size=action_expert_config.mlp_dim,
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="bfloat16",
                use_adarms=True,
                adarms_cond_dim=action_expert_config.width,
            )
            gemma_300m = GemmaForCausalLM(config=action_expert_config_hf)
            self.action_dit = copy.deepcopy(gemma_300m.model)
            del self.action_dit.embed_tokens
            self._default_pi05_action_dim = 32
            self.action_in_proj = nn.Linear(self._default_pi05_action_dim, action_expert_config.width)
            self.action_out_proj = nn.Linear(action_expert_config.width, self._default_pi05_action_dim)
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
            logging.info(f"Init Pi0.5 Action DiT Complete !")

        else:
            if getattr(self.config, "is_loc_learnable_query", False):
                self.loc_learnable_query = nn.Parameter(torch.randn(1, 1, llm_hidden_size))
            if getattr(self.config, "is_action_dit_projector", False):
                self.action_dit_projector = nn.Sequential(
                    nn.Linear(llm_hidden_size, llm_hidden_size*4, bias=True),
                    nn.GELU(),
                    nn.Linear(llm_hidden_size*4, llm_hidden_size*2, bias=True),
                    nn.GELU(),
                    nn.Linear(llm_hidden_size*2, llm_hidden_size, bias=True),
                )
                self.action_dit_norm = Qwen2RMSNorm(llm_hidden_size, eps=1e-6)
            path = model_args.mllm_hf_path
            logging.info(f"Initializing Action DiT from {path} slice...")
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=self.vision_tower.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            self.action_dit = copy.deepcopy(internvl_model.language_model)
            del self.action_dit.layers[:-getattr(model_args, "action_dit_layer", 3)]
            del self.action_dit.embed_tokens
            logging.info("Action DiT weights initialized successfully!")

            if getattr(model_args, 'is_action_dit_dense_timestep', False):
                # 替换 Norm 层为 AdaRMS
                logging.info("Replacing action_dit norms with AdaRMS...")
                # 1. 替换每一层的 Norm
                for layer in self.action_dit.layers:
                    # 替换 input_layernorm
                    old_norm = layer.input_layernorm
                    new_norm = Qwen2RMSNormAdaRMS(llm_hidden_size, cond_dim=llm_hidden_size, eps=old_norm.variance_epsilon)
                    new_norm.weight.data = old_norm.weight.data # 继承预训练权重
                    layer.input_layernorm = new_norm

                    # 替换 post_attention_layernorm
                    old_post_norm = layer.post_attention_layernorm
                    new_post_norm = Qwen2RMSNormAdaRMS(llm_hidden_size, cond_dim=llm_hidden_size, eps=old_post_norm.variance_epsilon)
                    new_post_norm.weight.data = old_post_norm.weight.data
                    layer.post_attention_layernorm = new_post_norm

                # 2. 替换 Final Norm
                old_final_norm = self.action_dit.norm
                new_final_norm = Qwen2RMSNormAdaRMS(llm_hidden_size, cond_dim=llm_hidden_size, eps=old_final_norm.variance_epsilon)
                new_final_norm.weight.data = old_final_norm.weight.data
                self.action_dit.norm = new_final_norm

            # Initialize Heads
            llm_hidden_size = self.config.text_config.hidden_size
            # self.action_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)#.to(torch.bfloat16)
            self.action_in_proj = nn.Linear(self.config.action_dim, llm_hidden_size)#.to(torch.bfloat16)
            self.time_mlp_in = nn.Linear(llm_hidden_size, llm_hidden_size)#.to(torch.bfloat16)
            self.time_mlp_out = nn.Linear(llm_hidden_size, llm_hidden_size)#.to(torch.bfloat16)
            self.action_out_proj = nn.Linear(llm_hidden_size, self.config.action_dim)#.to(torch.bfloat16)

        if  getattr(self.config, "use_codex_vit_regression_head", False):# vibe codex
            logging.info("Starting codex_vit_regression_head initialize...")
            for p in self.vit_loc_fusion.parameters(): p.requires_grad = True
            if getattr(self.config, "is_action_dit_projector", False):
                for p in self.action_dit_projector.parameters(): p.requires_grad = True
            for p in self.regression_loc_head.parameters(): p.requires_grad = True

            self.vit_loc_fusion.apply(init_weights)
            if getattr(self.config, "is_action_dit_projector", False):
                self.action_dit_projector.apply(init_weights)
            # self.action_dit_norm.apply(init_weights) # Norm层不需要手动初始化，保持原初始weights即可
            if getattr(self.config, "is_aciton_dit_vae_small_init", False):
                self.regression_loc_head.apply(small_init_weights)
            else:
                self.regression_loc_head.apply(init_weights)
            logging.info("codex_vit_regression_head weights initialized")
        elif  getattr(self.config, "use_vit_cls_regression_head", False):
            logging.info("Starting vit_cls_regression_head initialize...")
            if getattr(self.config, "is_action_dit_projector", False):
                for p in self.action_dit_projector.parameters(): p.requires_grad = True
            for p in self.regression_loc_head.parameters(): p.requires_grad = True

            if getattr(self.config, "is_action_dit_projector", False):
                self.action_dit_projector.apply(init_weights)
            # self.action_dit_norm.apply(init_weights) # Norm层不需要手动初始化，保持原初始weights即可
            if getattr(self.config, "is_aciton_dit_vae_small_init", False):
                self.regression_loc_head.apply(small_init_weights)
            else:
                self.regression_loc_head.apply(init_weights)
            logging.info("vit_cls_regression_head weights initialized")
        elif  getattr(self.config, "use_vit_regression_head", False):
            logging.info("Starting vit_regression_head initialize...")
            if getattr(self.config, "is_action_dit_projector", False):
                for p in self.action_dit_projector.parameters(): p.requires_grad = True
            for p in self.regression_loc_head.parameters(): p.requires_grad = True
            for p in self.cross_view_fusion.parameters(): p.requires_grad = True

            if getattr(self.config, "is_action_dit_projector", False):
                self.action_dit_projector.apply(init_weights)
            # self.action_dit_norm.apply(init_weights) # Norm层不需要手动初始化，保持原初始weights即可
            if getattr(self.config, "is_aciton_dit_vae_small_init", False):
                self.regression_loc_head.apply(small_init_weights)
            else:
                self.regression_loc_head.apply(init_weights)
            self.cross_view_fusion.apply(init_weights)
            logging.info("vit_regression_head weights initialized")
        else:
            ### 是否开启LoRA，重新配置可学习参数
            # 1. Action DiT (Pi0.5 GemmaExpertModel)
            self.is_lora = getattr(self.config, 'is_lora', False)
            if self.is_lora:
                # Backbone 冻结，等待 LoRA 注入
                for p in self.action_dit.parameters(): p.requires_grad = False
            else:
                for p in self.action_dit.parameters(): p.requires_grad = True
            # 2. Heads & Projectors
            # Enable Gradients for Action Path
            if getattr(self.config, "is_loc_learnable_query", False):
                for p in self.loc_learnable_query.parameters(): p.requires_grad = True
            if getattr(self.config, "is_action_dit_projector", False):
                for p in self.action_dit_projector.parameters(): p.requires_grad = True
                if getattr(self.config, "use_pi05_action_dit", False):
                    for p in self.action_dit_connector.parameters(): p.requires_grad = True
            for p in self.action_dit_norm.parameters(): p.requires_grad = True
            for p in self.action_in_proj.parameters(): p.requires_grad = True
            for p in self.time_mlp_in.parameters(): p.requires_grad = True
            for p in self.time_mlp_out.parameters(): p.requires_grad = True
            for p in self.action_out_proj.parameters(): p.requires_grad = True


            ### 直接移植pi05的action_dit模型和权重作为定位head,无需初始化权重
            if getattr(self.config, "use_pi05_action_dit", False):
                self.action_dit_connector.apply(init_weights)
                if getattr(self.config, "is_action_dit_projector", False):
                    self.action_dit_projector.apply(init_weights)
                    # self.action_dit_norm.apply(init_weights) # Norm层不需要手动初始化，保持原初始weights即可

                logging.info(f"Load Pi0.5 weights, from {self.config.pi05_pytorch_weight_path}")
                if os.path.exists(self.config.pi05_pytorch_weight_path):
                    model_path = os.path.join(self.config.pi05_pytorch_weight_path, "model.safetensors")
                else:
                    model_path = os.path.join("/home/user/yc57963/.cache/openpi/openpi-assets/checkpoints/pi05_base", "model.safetensors")
                # safetensors.torch.load_model(
                #     (self.action_dit.module if isinstance(self.action_dit, torch.nn.parallel.DistributedDataParallel) else self.action_dit), model_path, strict=False
                # )
                pi05_state_dict = safetensors.torch.load_file(model_path, device="cpu")
                self.action_dit_state_dict = self.action_dit.state_dict()

                new_action_dit_state_dict = {}
                new_action_in_proj_state_dict = {}
                new_action_out_proj_state_dict = {}
                new_time_mlp_in_state_dict = {}
                new_time_mlp_out_state_dict = {}
                # 打印层名映射日志
                logging.info("Starting pi05 weight conversion...")
                for key, value in pi05_state_dict.items():
                    new_key = None
                    if "action_in_proj" in key:
                        new_key = key.replace("action_in_proj.", "")
                        new_action_in_proj_state_dict[new_key] = value
                    if "action_out_proj" in key:
                        new_key = key.replace("action_out_proj.", "")
                        new_action_out_proj_state_dict[new_key] = value
                    if "time_mlp_in" in key:
                        new_key = key.replace("time_mlp_in.", "")
                        new_time_mlp_in_state_dict[new_key] = value
                    if "time_mlp_out" in key:
                        new_key = key.replace("time_mlp_out.", "")
                        new_time_mlp_out_state_dict[new_key] = value
                    if "paligemma_with_expert.gemma_expert.model." in key:
                        new_key = key.replace("paligemma_with_expert.gemma_expert.model.", "")
                        # if "dense." in new_key:
                        #     new_key = new_key.replace("dense.", "")
                        new_action_dit_state_dict[new_key] = value
                        if new_key in self.action_dit_state_dict and value.shape != self.action_dit_state_dict[new_key].shape:
                            logging.info(f"⚠️ Shape Mismatch for {new_key}: Source {value.shape} vs Target {self.action_dit_state_dict[new_key].shape}")

                msg = self.action_dit.load_state_dict(new_action_dit_state_dict, strict=False)
                logging.info(f"action_dit loading result: {msg}")
                msg = self.action_in_proj.load_state_dict(new_action_in_proj_state_dict, strict=False)
                logging.info(f"action_in_proj loading result: {msg}")
                msg = self.action_out_proj.load_state_dict(new_action_out_proj_state_dict, strict=False)
                logging.info(f"action_out_proj loading result: {msg}")
                msg = self.time_mlp_in.load_state_dict(new_time_mlp_in_state_dict, strict=False)
                logging.info(f"time_mlp_in loading result: {msg}")
                msg = self.time_mlp_out.load_state_dict(new_time_mlp_out_state_dict, strict=False)
                logging.info(f"time_mlp_out loading result: {msg}")
                print("Success: Loaded Action DiT weights from Pi0.5")
            else:
                if getattr(self.config, "is_action_dit_projector", False):
                    self.action_dit_projector.apply(init_weights)
                # self.action_dit_norm.apply(init_weights) # Norm层不需要手动初始化，保持原初始weights即可
                # Init Weights
                if getattr(self.config, "is_loc_learnable_query", False):
                    self.loc_learnable_query.apply(init_weights)
                if getattr(self.config, "is_aciton_dit_vae_small_init", False):
                    self.action_in_proj.apply(small_init_weights)
                else:
                    self.action_in_proj.apply(init_weights)
                self.time_mlp_in.apply(init_weights)
                self.time_mlp_out.apply(init_weights)
                if getattr(self.config, "is_aciton_dit_vae_small_init", False):
                    self.action_out_proj.apply(small_init_weights)
                else:
                    self.action_out_proj.apply(init_weights)
                logging.info(f"Custom Action DiT weights initialized, LoRA Enabled: {self.is_lora}")

            # if getattr(model_args, "gradient_checkpointing", False):
            #     self.action_dit.gradient_checkpointing_enable()


def split_image_tokens(input_ids, image_token_idx):
    # 1. 创建基础掩码：标记所有是图片 Token 的位置
    # Shape: [Batch, Seq_Len] (布尔值)
    all_img_mask = (input_ids == image_token_idx)

    # 2. 计算累加和：给每个图片 Token 编号 (1, 2, 3...)
    # 非图片位置虽然也有数值，但会被基础掩码过滤掉，所以不用担心
    # Shape: [Batch, Seq_Len]
    img_cumsum = all_img_mask.cumsum(dim=1)

    # 3. 计算每一行图片 Token 的总数
    # Shape: [Batch, 1] (保持维度以便广播)
    total_imgs = all_img_mask.sum(dim=1, keepdim=True)

    # 4. 计算分割点：总数的一半
    # Shape: [Batch, 1]
    half_point = total_imgs // 2

    # 5. 生成前一半的掩码 (und_image_idx)
    # 条件：是图片Token 且 当前累计序号 <= 一半
    und_image_idx = all_img_mask & (img_cumsum <= half_point)

    # 6. 生成后一半的掩码 (aux_image_idx)
    # 条件：是图片Token 且 当前累计序号 > 一半
    aux_image_idx = all_img_mask & (img_cumsum > half_point)

    return und_image_idx, aux_image_idx

class Unified_UniLIP_InternVL_MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().vision_tower

    def get_n_query(self):
        return self.get_model().config.n_query

    def _forward_vit_regression_head(self, und_image: Optional[torch.Tensor], aux_image: Optional[torch.Tensor] = None):
        vision_dtype = self.model.vision_tower.embeddings.patch_embedding.weight.dtype
        vision_tower = self.get_vision_tower()
        vision_feature_layer = getattr(self.config, "vision_feature_layer", -1)

        def _extract_vit_cls(pixel_values):
            if pixel_values is None:
                return None

            vision_outputs = vision_tower(
                pixel_values.to(dtype=vision_dtype),
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(vision_outputs, "hidden_states") and vision_outputs.hidden_states is not None:
                vit_hidden_states = vision_outputs.hidden_states[vision_feature_layer]
            else:
                vit_hidden_states = vision_outputs.last_hidden_state
            return vit_hidden_states[:, 0]

        if getattr(self.config, "is_action_dit_projector", False):
            target_dim = self.get_model().action_dit_norm.weight.shape[0]
        else:
            target_dim = self.get_model().regression_loc_head[0].in_features

        und_vit_cls_feature = _extract_vit_cls(und_image)
        aux_vit_cls_feature = _extract_vit_cls(aux_image)

        candidate_features = []
        if und_vit_cls_feature is not None and aux_vit_cls_feature is not None:
            candidate_features.extend([
                torch.cat([und_vit_cls_feature, aux_vit_cls_feature], dim=-1),
                und_vit_cls_feature + aux_vit_cls_feature,
                0.5 * (und_vit_cls_feature + aux_vit_cls_feature),
                und_vit_cls_feature,
                aux_vit_cls_feature,
            ])
        elif und_vit_cls_feature is not None:
            candidate_features.append(und_vit_cls_feature)
        elif aux_vit_cls_feature is not None:
            candidate_features.append(aux_vit_cls_feature)

        vit_feature = None
        for feature in candidate_features:
            if feature.shape[-1] == target_dim:
                vit_feature = feature
                break

        if vit_feature is None:
            raise ValueError(
                f"Cannot match ViT CLS feature dim to regression head input dim. "
                f"Target dim: {target_dim}, candidate dims: {[feature.shape[-1] for feature in candidate_features]}"
            )

        if getattr(self.config, "is_action_dit_projector", False):
            vit_feature = self.get_model().action_dit_norm(vit_feature)
            vit_feature = self.get_model().action_dit_projector(vit_feature)

        return self.get_model().regression_loc_head(vit_feature)

    # vibe codex
    def _forward_codex_vit_regression_head(self, und_image: Optional[torch.Tensor], aux_image: Optional[torch.Tensor] = None):
        vision_dtype = self.model.vision_tower.embeddings.patch_embedding.weight.dtype
        vision_tower = self.get_vision_tower()
        vision_feature_layer = getattr(self.config, "vision_feature_layer", -1)

        def _extract_codex_vit_features(pixel_values):
            if pixel_values is None:
                return None

            vision_outputs = vision_tower(
                pixel_values.to(dtype=vision_dtype),
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(vision_outputs, "hidden_states") and vision_outputs.hidden_states is not None:
                vit_hidden_states = vision_outputs.hidden_states[vision_feature_layer]
            else:
                vit_hidden_states = vision_outputs.last_hidden_state
            cls_feature = vit_hidden_states[:, 0]
            if vit_hidden_states.shape[1] > 1:
                pooled_feature = vit_hidden_states[:, 1:].mean(dim=1)
            else:
                pooled_feature = cls_feature
            return cls_feature, pooled_feature

        und_features = _extract_codex_vit_features(und_image)
        aux_features = _extract_codex_vit_features(aux_image)

        if und_features is None or aux_features is None:
            raise ValueError("Localization regression head requires both FPV and map images.")

        und_vit_cls_feature, und_vit_avg_feature = und_features
        aux_vit_cls_feature, aux_vit_avg_feature = aux_features

        if hasattr(self.get_model(), "vit_loc_fusion"):
            vit_feature = self.get_model().vit_loc_fusion(
                und_vit_cls_feature,
                aux_vit_cls_feature,
                und_vit_avg_feature,
                aux_vit_avg_feature,
            )
        else:
            vit_feature = torch.cat([und_vit_cls_feature, aux_vit_cls_feature], dim=-1)

        if getattr(self.config, "is_action_dit_projector", False):
            vit_feature = self.get_model().action_dit_norm(vit_feature)
            vit_feature = self.get_model().action_dit_projector(vit_feature)

        return self.get_model().regression_loc_head(vit_feature)

    # vibe codex
    def _compute_codex_loc_regression_loss(self, actions_pred, actions_gt):
        actions_pred = actions_pred.float()
        actions_gt = actions_gt.float()

        if actions_gt.ndim == 3 and actions_gt.shape[1] == 1:
            actions_gt = actions_gt.squeeze(1)
        if actions_pred.ndim == 3 and actions_pred.shape[1] == 1:
            actions_pred = actions_pred.squeeze(1)

        xy_loss = F.smooth_l1_loss(actions_pred[..., :2], actions_gt[..., :2], reduction="none").mean(dim=-1)
        z_loss = F.smooth_l1_loss(actions_pred[..., 2:3], actions_gt[..., 2:3], reduction="none").mean(dim=-1)

        if getattr(self.config, "loc_use_circular_loss", True):
            angle_delta = torch.remainder(actions_pred[..., 3:5] - actions_gt[..., 3:5] + 0.5, 1.0) - 0.5
            angle_loss = F.smooth_l1_loss(angle_delta, torch.zeros_like(angle_delta), reduction="none").mean(dim=-1)
        else:
            angle_loss = F.smooth_l1_loss(actions_pred[..., 3:5], actions_gt[..., 3:5], reduction="none").mean(dim=-1)

        xy_weight = getattr(self.config, "loc_xy_loss_weight", 1.0)
        z_weight = getattr(self.config, "loc_z_loss_weight", 1.0)
        angle_weight = getattr(self.config, "loc_angle_loss_weight", 2.0)

        total = xy_loss * xy_weight + z_loss * z_weight + angle_loss * angle_weight
        return total.mean()

    # [NEW] Flow Matching Logic Helper
    def embed_action_suffix(self, noisy_actions, timestep, llm_hidden_size, device, dtype):
        """
        Embed noisy_actions and timestep to prepare for Action Connector processing.
        Inspired by Pi0.5 embed_suffix but adapted for UniLIP architecture.
        """
        # 1. Embed Timestep (Sinusoidal -> MLP)
        # self.action_in_proj.out_features is llm_hidden_size
        time_emb = create_sinusoidal_pos_embedding(
            timestep, llm_hidden_size, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.to(dtype=dtype)

        # MLP Processing for Time (AdaRMS style condition)
        time_emb = self.get_model().time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.get_model().time_mlp_out(time_emb)
        adarms_cond = F.silu(time_emb) # [BS, Hidden]

        # 2. Embed Noisy Actions
        # noisy_actions: [BS, 1, Action_Dim]
        action_emb = self.get_model().action_in_proj(noisy_actions) # [BS, 1, Hidden]

        if not getattr(self.config, 'is_action_dit_dense_timestep', False):
            # 这里action_emb + adarms_cond，显式地将time传入action_embed，避免修改action_dit(internvl)的代码
            action_emb = action_emb + adarms_cond.unsqueeze(1)

        return action_emb, adarms_cond

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        # ... (Existing logic) ...
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gen_images, combined_und_images, grid_thw, i_s_pos, image_sizes,task_id
    ):
        # ... (Existing Logic: Process Images, Replace Tokens) ...
        # This part remains largely same as original code provided
        # Specifically handling und_images embedding replacement and latent_queries.

        #特征对齐：把输入的图片像素变成向量，并区分是“输入图（理解）”还是“目标图（生成）”。
        #嵌入替换：在文本 Embedding 序列中“挖坑”，把文本占位符 [IMG] 替换成真实的图片特征（对于理解任务）或可学习的 Query 向量（对于生成任务）。
        #训练目标设定：屏蔽 LLM 对图片位置的分类 Loss，准备好回归用的 Target Embeddings。

        # forward中使用 combined_und_images = torch.cat([und_image, aux_image], dim=0) 对齐传参格式
        und_images = combined_und_images[:combined_und_images.size(0)//2, ...] #torch.Size([128, 3, 448, 448])
        aux_images = combined_und_images[combined_und_images.size(0)//2:, ...] #torch.Size([128, 3, 448, 448])

        # [Simulated Abbreviation for Brevity - Keeping Core Logic]
        if (gen_images is None and und_images is None) or (aux_images is None and und_images is None) or input_ids.shape[1] == 1:
             return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None

        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy

        # 1. Process Gen Images (Target) - used for DiT Loss
        target_image_embeds = None
        if gen_images is not None and gen_images.sum() != 0: # Check if not empty (masked)
            with torch.no_grad():
                prompt_image_embeds = self.model.get_image_features(
                    pixel_values=gen_images,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_sizes=image_sizes,
                )# bs, 256, 896 # 为了对齐unilip预训练的latent_queries=256，gen_image仍使用448。因此有输入端：定位fps=224、map=224；生成map=224。输出端：生成fps=448
                # (B, HW, C) -> (B, C, H, W), assume H==W
                prompt_image_embeds = self.model.vae_decoder.clip_down(prompt_image_embeds)
            target_image_embeds = torch.clone(prompt_image_embeds).detach()
            target_image_embeds = target_image_embeds.mul_(self.model.unilip_factor) #torch.Size([128, 32, 16, 16]) # bs, 32, 8, 8

        # 2. Process Und Images (Input) - used for Understanding/Localization
        und_image_embeds = None
        und_img_idx = None
        if und_images is not None:
             und_image_embeds = self.model.get_image_features(
                pixel_values=und_images,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )#torch.Size([128, 256, 896])

        # 3. Process Aux Images (Input) - used for Understanding/Localization
        aux_image_embeds = None
        aux_img_idx = None
        if aux_images is not None:
             aux_image_embeds = self.model.get_image_features(
                pixel_values=aux_images,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )#torch.Size([128, 256, 896])

        # 4. Text Embeddings & Replacements
        und_image_idx, aux_image_idx = split_image_tokens(input_ids, IMAGE_TOKEN_IDX) # yiyong生成任务的最后也拼接了256个<IMG_CONTEXT> token #und_image_idx=torch.Size([128, 707]) #aux_image_idx=torch.Size([128, 707])
        gen_image_idx = (input_ids == IMAGE_TOKEN_IDX)
        # combined_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX) # 为了不改变原有模型的token_vocabulary, 直接使用UND_IMAGE_TOKEN_IDX作为und和aux image token的idx。
        text_embeds = self.get_model().language_model.embed_tokens(input_ids)

        # 5. Replace gen_token with Latent Queries
        is_gen_task = (task_id == 1)
        latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1) #torch.Size([128, 256, 896])
        valid_queries = latent_queries[is_gen_task] # 形状变更为 [N_Gen, 256, H] #赋值：现在 valid_queries.numel() == gen_img_idx.sum() * H，形状完美匹配
        H = valid_queries.shape[-1]
        valid_queries = valid_queries.contiguous().view(-1, H) #torch.Size([32768, 896])

        # Only replace if labels indicate generation task (output_indicator)
        output_indicator = labels != -100 #torch.Size([128, 707])
        gen_img_idx = torch.logical_and(output_indicator, gen_image_idx)
        text_embeds = text_embeds.clone() #torch.Size([128, 707, 896])
        if gen_img_idx.any(): #gen_img_idx.sum()=tensor(18176, device='cuda:0') #latent_queries=torch.Size([32768, 896]) #valid_queries.shape=torch.Size([18176, 896])
            text_embeds[gen_img_idx] = valid_queries.to(text_embeds.dtype)

        # 6. Replace Und & Aux Images
        is_loc_task = (task_id == 0)
        input_indicator = labels == -100
        und_img_idx = torch.logical_and(input_indicator, und_image_idx)
        aux_img_idx = torch.logical_and(input_indicator, aux_image_idx) #在生成任务中，利用split_image_tokens得到的aux_image_idx实际是gen_image_idx的位置，但labels!=-100，所以这里取交集aux_img_idx为空
        if und_images is not None and und_img_idx.any():
             text_embeds[und_img_idx] = und_image_embeds.to(text_embeds.device).flatten(0,1)
        if aux_images is not None and aux_img_idx.any():
             text_embeds[aux_img_idx] = aux_image_embeds[is_loc_task].to(text_embeds.device).flatten(0,1) #aux_image_embeds[is_loc_task]=[N_Loc, 256, H] #is_loc_task.sum()=tensor(57, device='cuda:0') #aux_img_idx.sum()=tensor(14592, device='cuda:0') #14592=57*256

        labels[gen_image_idx] = -100

        # Attention Mask for Bi-directional (if needed by Connector)
        bidr_attention_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)#attention_mask=torch.Size([128, 707])
        bidr_attention_mask = bidr_attention_mask.unsqueeze(1)
        bidr_attention_mask = (1-bidr_attention_mask.float())*-100000 #torch.Size([128, 1, 707, 707])

        combined_img_idx = torch.cat([und_img_idx, aux_img_idx], dim=0)#torch.Size([256, 707])
        combined_image_embeds = torch.cat([und_image_embeds, aux_image_embeds], dim=0)#torch.Size([256, 256, 896])
        return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds, combined_img_idx, combined_image_embeds, bidr_attention_mask

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token: #False
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end: #False
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token: #False
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


## transformers fix/lerobot_openpi branch
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            #self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate

        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        #self.dense.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

        # Apply adaptive normalization: use model weight dtype to ensure compatibility
        # model_dtype = self.dense.weight.dtype  # Use the model's dtype (bfloat16)
        # scale = scale.to(model_dtype)
        # shift = shift.to(model_dtype)
        # gate = gate.to(model_dtype)
        # normed_inputs = normed_inputs.to(model_dtype)  # Convert normed_inputs to model dtype

        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self):
        repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"
        if self.dense is not None:
            repr_str += f", adaptive=True, cond_dim={self.cond_dim}"
        return repr_str

# 将此类添加到 unified_unilip.py
class Qwen2RMSNormAdaRMS(nn.Module):
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

        # [修改] 输出维度变为 3倍 (Scale, Shift, Gate)
        self.linear = nn.Linear(cond_dim, hidden_size * 3)

        # 零初始化 (保持 AdaLN-Zero 的特性)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states, cond):
        # 1. 基础 RMSNorm 计算 (不带仿射变换)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)#torch.Size([2, 708, 896])
        variance = hidden_states.pow(2).mean(-1, keepdim=True)#torch.Size([2, 708, 1])
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 使用预训练的 weight 进行缩放 (保留原模型知识)
        normed = self.weight * hidden_states.to(input_dtype)

        # 2. AdaRMS 调制
        # 投影: [BS, Cond_Dim] -> [BS, 3 * Hidden]
        modulation = self.linear(cond)#torch.Size([2, 2688])
        modulation = modulation.unsqueeze(1) # [BS, 1, 3 * Hidden]

        # 切分: Scale, Shift, Gate
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)#3 * torch.Size([2, 1, 896])

        # 应用: Norm(x) * (1 + scale) + shift
        # 即使是 RMSNorm，在 Diffusion 中也习惯加上 Shift 以增强表达能力
        normed = normed.to(input_dtype) * (1 + scale) + shift

        # [关键] 返回 归一化后的特征 和 门控值
        return normed, gate

# class Qwen2RMSNormAdaRMS(nn.Module):
#     def __init__(self, hidden_size, cond_dim, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#         # [AdaRMS 核心] 条件投影层
#         # 将 cond (时间步特征) 映射为缩放系数
#         self.linear = nn.Linear(cond_dim, hidden_size)

#         # [关键] 零初始化
#         # 确保初始状态下 scale=0，输出等于原始 RMSNorm，不破坏预训练权重
#         nn.init.zeros_(self.linear.weight)
#         nn.init.zeros_(self.linear.bias)

#     def forward(self, hidden_states, cond):
#         # 1. 标准 RMSNorm 计算
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         normed = self.weight * hidden_states.to(input_dtype)

#         # 2. AdaRMS 调制
#         # cond: [BS, Cond_Dim] -> scale: [BS, Hidden]
#         scale = self.linear(cond)

#         # 广播维度: [BS, Hidden] -> [BS, 1, Hidden] 以匹配序列长度
#         scale = scale.unsqueeze(1)

#         # 调制: Norm(x) * (1 + scale)
#         return normed * (1 + scale)


# ==========================================
# 2. Main Model Class (InternVL Structure)
# ==========================================
class Unified_UniLIP_InternVLConfig(InternVLConfig):
    model_type = "unified_unilip"

class Unified_UniLIP_InternVLModel(Unified_UniLIP_InternVL_MetaModel, InternVLModel):
    config_class = Unified_UniLIP_InternVLConfig
    def __init__(self, config: InternVLConfig):
        super(Unified_UniLIP_InternVLModel, self).__init__(config)

class Unified_UniLIP_InternVLForCausalLM(InternVLForConditionalGeneration, Unified_UniLIP_InternVL_MetaForCausalLM):
    config_class = Unified_UniLIP_InternVLConfig

    def __init__(self, config):
        InternVLForConditionalGeneration.__init__(self, config)
        config.model_type = "unified_unilip"
        self.model = Unified_UniLIP_InternVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    # 通用 LoRA 注入函数
    def _apply_lora_to_module(self, lora_r, lora_alpha, lora_dropout, target_modules, modules_to_save=None, module_name="submodule"):
        """
        Helper to inject LoRA adapters into specific sub-modules.
        """
        # 1. 检查模块是否存在
        if not hasattr(self.model, module_name):
            logging.warning(f"⚠️ Module {module_name} not found in model, skipping LoRA injection.")
            return
        # 2. 获取子模块对象
        module = getattr(self.model, module_name)
        logging.info(f"🚀 Injecting LoRA into sub-module: {module_name}...")


        # # =================================================================
        # # [关键修复] 强制打补丁 (Force Monkey Patch)
        # # =================================================================
        # # 针对 action_dit 和 llm_connector 这种llm slices 删除了 embedding 的模块，
        # # 无论它们是否开启 Gradient Checkpointing，都强制给一个假的 get_input_embeddings。
        # # 这样可以一劳永逸地解决 PEFT 的自动检查报错。
        if module_name in ["llm_connector", "action_dit"]:
            import types
            def _get_input_embeddings_shim(self_obj):
                if not hasattr(self_obj, "_dummy_embedding"):
                    self_obj._dummy_embedding = torch.nn.Identity().to(self_obj.device)
                    self_obj._dummy_embedding.weight = torch.tensor([0.0], requires_grad=True, device=self_obj.device)
                return self_obj._dummy_embedding

            # 强制替换实例方法 (不要做任何检查，直接覆盖！)
            logging.info(f"🔧 Patching get_input_embeddings for {module_name} to bypass PEFT check.")
            module.get_input_embeddings = types.MethodType(_get_input_embeddings_shim, module)


        # 3. 配置 LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None, # 对于子模块通常不需要指定 TaskType，作为通用 Module 处理
            modules_to_save=modules_to_save # 投影层我们手动设置 requires_grad
        )
        logging.info(f"Applying LoRA to {module_name} with target_modules: {target_modules}, modules_to_save: {modules_to_save}")
        # 4. 包装并原地替换
        peft_module = get_peft_model(module, lora_config)
        setattr(self.model, module_name, peft_module)
        # 5. 打印可训练参数量以验证
        logging.info(f"📊 {module_name} Adapter Config:")
        peft_module.print_trainable_parameters()

        # 6. [Hack] 确保 modules_to_save 中的参数 requires_grad=True
        # 有时 get_peft_model 对自定义嵌套模块的 modules_to_save 处理不完美
        if modules_to_save is not None:
            for name, param in self.model.named_parameters():
                if any(m in name for m in modules_to_save):
                    param.requires_grad = True

        return peft_module

    def _get_shared_llm_tail_layer_indices(self):
        total_layers = len(self.model.language_model.layers)
        tail_num_layers = min(
            max(int(getattr(self.config, "shared_llm_tail_num_layers", 0) or 0), 0),
            total_layers,
        )
        start_layer_idx = max(0, total_layers - tail_num_layers)
        return list(range(start_layer_idx, total_layers))

    def _apply_lora_to_language_model_tail_layers(self, model_args, training_args):
        tail_layer_indices = self._get_shared_llm_tail_layer_indices()
        if len(tail_layer_indices) == 0:
            logging.warning("shared_llm_tail_lora_enabled=True but tail layer range is empty, skipping tail LoRA injection.")
            return

        logging.info(
            "🚀 Injecting dedicated tail LoRA into language_model.layers[%d:%d) with mode=%s",
            tail_layer_indices[0],
            tail_layer_indices[-1] + 1,
            getattr(model_args, "shared_llm_tail_lora_mode", "lora_only"),
        )

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        for layer_idx in tail_layer_indices:
            layer_module = self.model.language_model.layers[layer_idx]
            if isinstance(layer_module, PeftModel) or hasattr(layer_module, "peft_config"):
                raise RuntimeError(
                    f"language_model.layers[{layer_idx}] is already wrapped by PEFT; refusing to inject shared tail LoRA twice."
                )

            lora_config = LoraConfig(
                r=training_args.shared_llm_tail_lora_r,
                lora_alpha=training_args.shared_llm_tail_lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.shared_llm_tail_lora_dropout,
                bias="none",
                task_type=None,
                modules_to_save=None,
            )
            wrapped_layer = get_peft_model(layer_module, lora_config)
            self.model.language_model.layers[layer_idx] = wrapped_layer
            logging.info("📊 language_model.layers.%d dedicated tail LoRA injected", layer_idx)
            wrapped_layer.print_trainable_parameters()

    def _freeze_language_model_tail_base_keep_lora_only(self):
        tail_layer_indices = self._get_shared_llm_tail_layer_indices()
        for layer_idx in tail_layer_indices:
            layer_module = self.model.language_model.layers[layer_idx]
            for name, param in layer_module.named_parameters():
                param.requires_grad = ("lora_" in name)

        logging.info(
            "🔒 Applied lora_only freeze to language_model tail layers [%d:%d)",
            tail_layer_indices[0] if len(tail_layer_indices) > 0 else 0,
            (tail_layer_indices[-1] + 1) if len(tail_layer_indices) > 0 else 0,
        )

    def inject_lora_to_sub_module(self, model_args, training_args):
        if not getattr(training_args, 'is_lora', False) and not getattr(model_args, "shared_llm_tail_lora_enabled", False):
            return

        logging.info("🌟 Starting Modular LoRA Injection for Unified UniLIP...")

        # =========================================================
        # 1. Vision Tower (InternVisionModel)
        # =========================================================
        # 结构: attn.qkv, attn.proj, mlp.fc1, mlp.fc2
        if not model_args.fix_vit and not getattr(model_args, "train_mm_projector_only", False):
            self._apply_lora_to_module(
                lora_r=training_args.lora_r // 2,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=["qkv", "proj", "fc1", "fc2",], #["qkv", "proj", "fc1", "fc2"]
                # modules_to_save=["multi_modal_projector"],
                module_name="vision_tower"
            )
        # =========================================================
        # 2. LLM Backbone (Qwen2Model)
        # =========================================================
        # 结构: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        if not model_args.fix_llm and not getattr(model_args, "train_shared_llm_tail_only", False):
            self._apply_lora_to_module(
                lora_r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                # modules_to_save=["lm_head", "embed_tokens"],
                module_name="language_model"
            )
        elif getattr(model_args, "train_shared_llm_tail_only", False) and getattr(model_args, "shared_llm_tail_lora_enabled", False):
            self._apply_lora_to_language_model_tail_layers(model_args, training_args)
            if getattr(model_args, "shared_llm_tail_lora_mode", "lora_only") == "lora_only":
                self._freeze_language_model_tail_base_keep_lora_only()
        # =========================================================
        # 3. LLM Connector (Qwen2Model Slice)
        # =========================================================
        # 结构同 LLM
        if getattr(training_args, 'is_lora', False) and not self.get_model().fix_connect:
            self._apply_lora_to_module(
                lora_r=training_args.lora_r // 2,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                # modules_to_save=None,
                module_name="llm_connector"
            )
        # =========================================================
        # 4. Gen DiT (SanaTransformer2DModel)
        # =========================================================
        # 结构分析:
        # attn1/attn2: to_q, to_k, to_v, to_out.0
        # PatchEmbed/Timestep: linear_1, linear_2
        # 注意：Sana 的 GLUMBConv 使用的是 Conv2d，LoRA 默认不转 Conv2d 除非指定。
        # 这里我们主要对 Attention 和 Timestep MLP 做 LoRA。
        # to_q/k/v 匹配 Attention, linear_1/2 匹配 TimestepEmbedder & CaptionProjection
        if getattr(training_args, 'is_lora', False) and not self.get_model().fix_dit:
            self._apply_lora_to_module(
                lora_r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=["to_q", "to_k", "to_v", "to_out.0", "linear_1", "linear_2"],
                # modules_to_save=None,
                module_name="dit"
            )
        # =========================================================
        # 5. Loc Action DiT (Qwen2Model Slice) or (Pi05 Action DiT)
        # =========================================================
        # 结构同 LLM
        if getattr(training_args, 'is_lora', False):
            self._apply_lora_to_module(
                lora_r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                # modules_to_save=None,
                module_name="action_dit"
            )

        # =========================================================
        # 6. [关键] 统一开启非 LoRA 模块 (Heads/Projectors) 的梯度
        # =========================================================
        # 默认 modules_to_save=None (因为它是针对子模块内的)，所以需要手动开启外部连接层的梯度。

        logging.info("🔓 Unfreezing Projectors and Heads...")

        # 定义需要全量训练的模块关键词
        modules_to_train_fully = [
            # "lm_head",                # LLM Output
            # "embed_tokens",           # LLM Input Embedding (如果 resize 了)
            # "projector",              # Connector -> DiT
            "latent_queries",         # Gen Query
            "action_dit_connector",
            "action_dit_projector",   # LLM -> Action DiT
            "action_dit_norm",        # Action DiT Norm (AdaRMS)
            "action_in_proj",         # Action Input
            "action_out_proj",        # Action Output
            "time_mlp_in",            # Timestep MLP
            "time_mlp_out",
            "loc_learnable_query",
            "vit_regression_head",
            "vit_cls_regression_head",
        ]
        if getattr(model_args, "train_mm_projector_only", False):
            modules_to_train_fully.append("multi_modal_projector")

        count_unfrozen = 0
        for name, param in self.model.named_parameters():
            # 检查参数名是否包含上述关键词
            if any(m in name for m in modules_to_train_fully):
                if param.requires_grad == False:
                    logging.info(name)
                    param.requires_grad = True
                    count_unfrozen += 1

        logging.info(f"✅ LoRA Injection Complete. Manually unfroze {count_unfrozen} parameters for Heads/Projectors.")

    def _prepare_attention_masks_4d_from_attn_masks_1d(self, att_masks_1d):
        # 手动构建 4D Causal Mask
        batch_size, seq_len = att_masks_1d.shape[:2]

        # 1. 创建下三角 Causal Mask [Seq, Seq]
        # min_dtype 是 float 的最小值 (e.g. -65504 for fp16, -3.4e38 for fp32)
        min_dtype = torch.finfo(self.get_model().dtype).min
        causal_mask = torch.full((seq_len, seq_len), min_dtype, device=att_masks_1d.device, dtype=self.get_model().dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1) # 上三角为负无穷，下三角为0

        # 2. 扩展维度 [1, 1, Seq, Seq]
        causal_mask = causal_mask[None, None, :, :]

        # 3. 处理 Padding Mask [BS, Seq] -> [BS, 1, 1, Seq]
        # 注意：attention_mask 是 1=Valid, 0=Pad
        # 我们需要把 0 变成负无穷，1 变成 0
        padding_mask = torch.zeros_like(att_masks_1d, dtype=self.get_model().dtype)
        padding_mask = padding_mask.masked_fill(att_masks_1d == 0, min_dtype)
        padding_mask = padding_mask[:, None, None, :]

        # 4. 合并 [BS, 1, Seq, Seq]
        # 利用广播机制：Causal (mask future) + Padding (mask pad tokens)
        combined_mask = causal_mask + padding_mask
        return combined_mask

    def get_model(self):
        return self.model

    def initialize_repa_modules_from_config(self) -> None:
        if not getattr(self.config, "is_repa_loss", False):
            return

        projector_type = getattr(self.config, "repa_projector_type", "mlp3_silu")
        if projector_type not in {"mlp3_silu", "conv_spatialnorm"}:
            raise ValueError(f"Unsupported repa_projector_type for current implementation: {projector_type}")

        if hasattr(self.model, "repa_projector"):
            return

        student_dim = self.get_model().dit.config.num_attention_heads * self.get_model().dit.config.attention_head_dim
        teacher_type = getattr(self.config, "repa_teacher_type", "dinov2")
        if teacher_type not in {"dinov2", "unilip_vision"}:
            raise ValueError(f"Unsupported repa_teacher_type for current implementation: {teacher_type}")

        teacher_dim = int(getattr(self.config, "repa_teacher_hidden_size", 768))
        if projector_type == "mlp3_silu":
            num_layers = int(getattr(self.config, "repa_mlp_num_layers", 3))
            hidden_ratio = float(getattr(self.config, "repa_mlp_hidden_ratio", 1.0))
            self.model.repa_projector = REPAMLPProjector(
                in_dim=student_dim,
                out_dim=teacher_dim,
                num_layers=num_layers,
                hidden_ratio=hidden_ratio,
            )
            projector_log_kwargs = {
                "num_layers": num_layers,
                "hidden_ratio": hidden_ratio,
            }
        else:
            expected_num_patches = int(getattr(self.config, "repa_expected_num_patches", 256))
            conv_kernel_size = int(getattr(self.config, "repa_conv_kernel_size", 3))
            use_spatial_norm = bool(getattr(self.config, "repa_use_spatial_norm", True))
            spatial_norm_gamma = float(getattr(self.config, "repa_spatial_norm_gamma", 1.0))
            self.model.repa_projector = REPAConvSpatialNormProjector(
                in_dim=student_dim,
                out_dim=teacher_dim,
                expected_num_patches=expected_num_patches,
                kernel_size=conv_kernel_size,
                use_spatial_norm=use_spatial_norm,
                spatial_norm_gamma=spatial_norm_gamma,
            )
            projector_log_kwargs = {
                "expected_num_patches": expected_num_patches,
                "kernel_size": conv_kernel_size,
                "use_spatial_norm": use_spatial_norm,
                "spatial_norm_gamma": spatial_norm_gamma,
            }
        self.model.repa_projector.to(dtype=self.get_model().projector.weight.dtype)
        self.model.repa_projector.requires_grad_(True)
        logging.info(
            "Initialized REPA projector: type=%s student_dim=%s teacher_dim=%s extra=%s",
            projector_type,
            student_dim,
            teacher_dim,
            projector_log_kwargs,
        )

    def set_repa_teacher(self, teacher_bundle: Dict[str, nn.Module]) -> None:
        self.__dict__["_repa_teacher"] = teacher_bundle

    def get_repa_teacher(self):
        return self.__dict__.get("_repa_teacher", None)

    def _ensure_repa_teacher_on_device(self, device: torch.device) -> None:
        teacher_bundle = self.get_repa_teacher()
        if teacher_bundle is None:
            return

        teacher_type = teacher_bundle.get("type", "dinov2")
        if teacher_type == "dinov2":
            teacher_model = teacher_bundle["model"]
            try:
                teacher_param = next(teacher_model.parameters())
                teacher_device = teacher_param.device
            except StopIteration:
                teacher_device = device

            if teacher_device != device:
                teacher_model.to(device=device)
            return

        if teacher_type == "unilip_vision":
            for module_name in ["vision_tower", "multi_modal_projector"]:
                teacher_module = teacher_bundle[module_name]
                ref_module = getattr(self.model, module_name)
                try:
                    ref_param = next(ref_module.parameters())
                    target_dtype = ref_param.dtype
                except StopIteration:
                    target_dtype = None

                try:
                    teacher_param = next(teacher_module.parameters())
                    teacher_device = teacher_param.device
                    teacher_dtype = teacher_param.dtype
                except StopIteration:
                    teacher_device = device
                    teacher_dtype = target_dtype

                if teacher_device != device or (target_dtype is not None and teacher_dtype != target_dtype):
                    if target_dtype is not None:
                        teacher_module.to(device=device, dtype=target_dtype)
                    else:
                        teacher_module.to(device=device)
            return

        raise ValueError(f"Unsupported repa_teacher_type for device placement: {teacher_type}")

    def initialize_repa_teacher(self) -> None:
        teacher_type = getattr(self.config, "repa_teacher_type", "dinov2")
        teacher_name = getattr(self.config, "repa_teacher_name_or_path", "facebook/dinov2-base")

        if teacher_type == "dinov2":
            teacher_model = AutoModel.from_pretrained(
                teacher_name,
                low_cpu_mem_usage=True,
            )
            teacher_model.eval()
            teacher_model.requires_grad_(False)

            teacher_hidden_size = int(getattr(teacher_model.config, "hidden_size", 768))
            self.config.repa_teacher_hidden_size = teacher_hidden_size
            self.set_repa_teacher(
                {
                    "type": teacher_type,
                    "name_or_path": teacher_name,
                    "model": teacher_model,
                }
            )
            logging.info(
                "REPA teacher attached (frozen eval): type=%s, name_or_path=%s, hidden_size=%s",
                teacher_type,
                teacher_name,
                teacher_hidden_size,
            )
            return

        if teacher_type == "unilip_vision":
            vision_tower = copy.deepcopy(self.model.vision_tower)
            multi_modal_projector = copy.deepcopy(self.model.multi_modal_projector)
            vision_tower.eval()
            vision_tower.requires_grad_(False)
            multi_modal_projector.eval()
            multi_modal_projector.requires_grad_(False)

            teacher_hidden_size = int(getattr(self.config, "repa_teacher_hidden_size", 0))
            if teacher_hidden_size <= 0:
                projector_tail = multi_modal_projector[-1] if isinstance(multi_modal_projector, nn.Sequential) else multi_modal_projector
                teacher_hidden_size = getattr(projector_tail, "out_features", None)
            if teacher_hidden_size is None or int(teacher_hidden_size) <= 0:
                raise ValueError("repa_teacher_hidden_size must be set for repa_teacher_type='unilip_vision'.")

            self.config.repa_teacher_hidden_size = int(teacher_hidden_size)
            self.set_repa_teacher(
                {
                    "type": teacher_type,
                    "name_or_path": teacher_name,
                    "vision_tower": vision_tower,
                    "multi_modal_projector": multi_modal_projector,
                }
            )
            logging.info(
                "REPA teacher attached (frozen eval): type=%s, name_or_path=%s, hidden_size=%s",
                teacher_type,
                teacher_name,
                teacher_hidden_size,
            )
            return

        raise ValueError(f"Unsupported repa_teacher_type for current implementation: {teacher_type}")

    def set_external_localization_model(self, external_model: nn.Module) -> None:
        # Keep it out of nn.Module registration to avoid saving/updating in UniLIP checkpoints.
        self.__dict__["_external_localization_model"] = external_model

    def get_external_localization_model(self):
        return self.__dict__.get("_external_localization_model", None)

    def set_loc_repa_teacher(self, teacher_bundle: Dict[str, nn.Module]) -> None:
        self.__dict__["_loc_repa_teacher"] = teacher_bundle

    def get_loc_repa_teacher(self):
        return self.__dict__.get("_loc_repa_teacher", None)

    def _ensure_loc_repa_teacher_on_device(self, device: torch.device) -> None:
        teacher_bundle = self.get_loc_repa_teacher()
        if teacher_bundle is None:
            return

        module_names = [
            "vision_tower",
            "multi_modal_projector",
            "language_model",
            "action_dit_connector",
            "action_dit_norm",
            "action_dit_projector",
        ]
        for module_name in module_names:
            teacher_module = teacher_bundle[module_name]
            student_module = getattr(self.model, module_name)
            try:
                ref_param = next(student_module.parameters())
                target_dtype = ref_param.dtype
            except StopIteration:
                target_dtype = None

            try:
                teacher_param = next(teacher_module.parameters())
                teacher_device = teacher_param.device
                teacher_dtype = teacher_param.dtype
            except StopIteration:
                teacher_device = device
                teacher_dtype = target_dtype

            if teacher_device != device or (target_dtype is not None and teacher_dtype != target_dtype):
                if target_dtype is not None:
                    teacher_module.to(device=device, dtype=target_dtype)
                else:
                    teacher_module.to(device=device)

    def initialize_loc_repa_teacher_from_state_dict(self, teacher_state_dict: Dict[str, torch.Tensor]) -> None:
        if not getattr(self.config, "use_pi05_action_dit", False):
            raise ValueError("loc_repa teacher currently only supports use_pi05_action_dit=True.")
        if not getattr(self.config, "is_action_dit_projector", False):
            raise ValueError("loc_repa teacher currently requires is_action_dit_projector=True.")
        if not hasattr(self.model, "action_dit_connector"):
            raise RuntimeError("Current model has no action_dit_connector; cannot initialize loc_repa teacher.")

        teacher_bundle = {
            "vision_tower": copy.deepcopy(self.model.vision_tower),
            "multi_modal_projector": copy.deepcopy(self.model.multi_modal_projector),
            "language_model": copy.deepcopy(self.model.language_model),
            "action_dit_connector": copy.deepcopy(self.model.action_dit_connector),
            "action_dit_norm": copy.deepcopy(self.model.action_dit_norm),
            "action_dit_projector": copy.deepcopy(self.model.action_dit_projector),
        }

        teacher_modules = [
            ("vision_tower", teacher_bundle["vision_tower"]),
            ("multi_modal_projector", teacher_bundle["multi_modal_projector"]),
            ("language_model", teacher_bundle["language_model"]),
            ("action_dit_connector", teacher_bundle["action_dit_connector"]),
            ("action_dit_norm", teacher_bundle["action_dit_norm"]),
            ("action_dit_projector", teacher_bundle["action_dit_projector"]),
        ]

        for prefix, module in teacher_modules:
            state_to_load = {}
            full_prefix = f"model.{prefix}."
            for key, value in teacher_state_dict.items():
                if key == f"model.{prefix}":
                    continue
                if key.startswith(full_prefix):
                    state_to_load[key[len(full_prefix):]] = value

            if len(state_to_load) == 0:
                raise RuntimeError(f"Failed to find any teacher weights for {prefix} in loc_repa teacher checkpoint.")

            msg = module.load_state_dict(state_to_load, strict=False)
            logging.info("Loaded loc_repa teacher %s: %s", prefix, msg)
            ref_module = getattr(self.model, prefix)
            try:
                ref_param = next(ref_module.parameters())
                module.to(device=ref_param.device, dtype=ref_param.dtype)
            except StopIteration:
                module.to(device=self.device, dtype=self.dtype)
            module.eval()
            module.requires_grad_(False)

        self.set_loc_repa_teacher(teacher_bundle)

    def _decode_pred_pixels_input_from_flow(
        self,
        sigmas: torch.Tensor,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        und_image_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_latents_x0 = noisy_latents - sigmas * noise_pred
        pred_latents_scaled = pred_latents_x0 / self.model.config.unilip_factor

        with torch.enable_grad():
            mini_batch_size = 64
            pred_pixels_list = []
            for i in range(0, pred_latents_scaled.shape[0], mini_batch_size):
                batch_latents = pred_latents_scaled[i : i + mini_batch_size]
                batch_pixels = self.model.vae_decoder.vae_decode(batch_latents)
                pred_pixels_list.append(batch_pixels)
            pred_pixels = torch.cat(pred_pixels_list, dim=0)

        pred_pixels_norm = (pred_pixels + 1.0) / 2.0
        target_hw = und_image_map.shape[-2:]
        if pred_pixels_norm.shape[-2:] != target_hw:
            pred_pixels_norm = F.interpolate(
                pred_pixels_norm,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        pred_pixels_input = (pred_pixels_norm - 0.5) / 0.5
        return pred_pixels_input, pred_pixels_norm

    def _pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _extract_teacher_image_embeds(self, teacher_bundle, pixel_values):
        if pixel_values is None:
            return None
        vision_tower = teacher_bundle["vision_tower"]
        multi_modal_projector = teacher_bundle["multi_modal_projector"]
        vision_dtype = vision_tower.embeddings.patch_embedding.weight.dtype
        vision_feature_layer = getattr(self.config, "vision_feature_layer", -1)
        if vision_feature_layer == -1:
            vit_embeds = vision_tower(
                pixel_values.to(dtype=vision_dtype),
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = vision_tower(
                pixel_values.to(dtype=vision_dtype),
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[vision_feature_layer]
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self._pixel_shuffle(vit_embeds, scale_factor=0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = multi_modal_projector(vit_embeds)
        return vit_embeds

    def _normalize_for_repa_teacher(self, image: torch.Tensor) -> torch.Tensor:
        image_01 = (image.float() + 1.0) / 2.0
        target_size = int(getattr(self.config, "repa_teacher_input_size", 224))
        if image_01.shape[-2:] != (target_size, target_size):
            image_01 = F.interpolate(
                image_01,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = torch.tensor([0.485, 0.456, 0.406], device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
        return (image_01 - mean) / std

    def _extract_repa_teacher_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        teacher_bundle = self.get_repa_teacher()
        if teacher_bundle is None:
            raise RuntimeError("REPA teacher is not initialized before teacher feature extraction.")

        self._ensure_repa_teacher_on_device(image.device)
        teacher_type = teacher_bundle.get("type", "dinov2")
        if teacher_type == "dinov2":
            teacher_model = teacher_bundle["model"]
            teacher_dtype = next(teacher_model.parameters()).dtype
            teacher_inputs = self._normalize_for_repa_teacher(image).to(device=image.device, dtype=teacher_dtype)

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    pixel_values=teacher_inputs,
                    output_hidden_states=False,
                    return_dict=True,
                )

            teacher_feat = teacher_outputs.last_hidden_state[:, 1:, :]
        elif teacher_type == "unilip_vision":
            with torch.no_grad():
                teacher_feat = self._extract_teacher_image_embeds(teacher_bundle, image)
        else:
            raise ValueError(f"Unsupported repa_teacher_type for teacher feature extraction: {teacher_type}")

        expected_tokens = int(getattr(self.config, "repa_expected_num_patches", 256))
        if teacher_feat.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Unexpected REPA teacher patch count: got {teacher_feat.shape[1]}, expected {expected_tokens}."
            )
        return teacher_feat

    def _forward_dit_with_repa_hidden(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        detach_condition: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        block_idx = int(getattr(self.config, "repa_dit_layer_idx", 6))
        dit_model = self.get_model().dit
        if not hasattr(dit_model, "transformer_blocks"):
            raise RuntimeError("Current DiT model has no transformer_blocks; cannot capture REPA hidden states.")
        if block_idx < 0 or block_idx >= len(dit_model.transformer_blocks):
            raise ValueError(f"repa_dit_layer_idx={block_idx} out of range for {len(dit_model.transformer_blocks)} blocks")

        hidden_cache = {}

        def _capture_hidden(_module, _inputs, output):
            hidden_cache["hidden"] = output

        handle = dit_model.transformer_blocks[block_idx].register_forward_hook(_capture_hidden)
        try:
            dit_encoder_hidden_states = encoder_hidden_states.detach() if detach_condition else encoder_hidden_states
            noise_pred = dit_model(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=dit_encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        finally:
            handle.remove()

        repa_hidden = hidden_cache.get("hidden", None)
        if repa_hidden is None:
            raise RuntimeError(f"Failed to capture REPA hidden states from DiT block {block_idx}.")
        return noise_pred, repa_hidden

    def _extract_repa_student_patch_features(self, repa_hidden: torch.Tensor) -> torch.Tensor:
        align_type = getattr(self.config, "repa_align_type", "patch_wise")
        if align_type != "patch_wise":
            raise ValueError(f"Unsupported repa_align_type for current implementation: {align_type}")
        if not hasattr(self.get_model(), "repa_projector"):
            raise RuntimeError("REPA projector is not initialized before student feature extraction.")

        expected_tokens = int(getattr(self.config, "repa_expected_num_patches", 256))
        if repa_hidden.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Unexpected REPA student token count: got {repa_hidden.shape[1]}, expected {expected_tokens}."
            )

        student_feat = self.get_model().repa_projector(repa_hidden)
        return student_feat

    def _compute_repa_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        student_feat = F.normalize(student_feat.float(), dim=-1)
        teacher_feat = F.normalize(teacher_feat.detach().float(), dim=-1)
        token_loss = 1.0 - (student_feat * teacher_feat).sum(dim=-1)
        sample_loss = token_loss.mean(dim=-1)
        return (sample_loss * loss_mask[:, 1].float()).mean().to(torch.float32)

    def forward_for_repa_loss(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        clean_image: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, repa_hidden = self._forward_dit_with_repa_hidden(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            detach_condition=bool(getattr(self.config, "repa_detach_condition", True)),
        )
        student_feat = self._extract_repa_student_patch_features(repa_hidden)
        teacher_feat = self._extract_repa_teacher_patch_features(clean_image)
        return self._compute_repa_loss(student_feat, teacher_feat, loss_mask)

    def _extract_loc_repa_prefix_features(
        self,
        fps_image: torch.Tensor,
        map_image: torch.Tensor,
        loc_input_ids: torch.Tensor,
        loc_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        task_id: torch.Tensor,
        use_teacher: bool = False,
    ) -> torch.Tensor:
        if not getattr(self.config, "use_pi05_action_dit", False):
            raise ValueError("loc_repa feature extraction currently only supports use_pi05_action_dit=True.")
        if not getattr(self.config, "is_action_dit_projector", False):
            raise ValueError("loc_repa feature extraction currently requires is_action_dit_projector=True.")
        if not getattr(self.config, "loc_repa_use_und_tokens_only", True):
            raise ValueError("loc_repa currently only supports loc_repa_use_und_tokens_only=True.")

        if use_teacher:
            teacher_bundle = self.get_loc_repa_teacher()
            if teacher_bundle is None:
                raise RuntimeError("loc_repa teacher is not initialized before teacher feature extraction.")
            self._ensure_loc_repa_teacher_on_device(fps_image.device)

            with torch.no_grad():
                teacher_language_model = teacher_bundle["language_model"]
                und_image_embeds = self._extract_teacher_image_embeds(teacher_bundle, fps_image)
                aux_image_embeds = self._extract_teacher_image_embeds(teacher_bundle, map_image)

                und_image_idx, aux_image_idx = split_image_tokens(loc_input_ids, IMAGE_TOKEN_IDX)
                text_embeds = teacher_language_model.embed_tokens(loc_input_ids)
                text_embeds = text_embeds.clone()
                if und_image_embeds is not None and und_image_idx.any():
                    text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device).flatten(0, 1)
                if aux_image_embeds is not None and aux_image_idx.any():
                    text_embeds[aux_image_idx] = aux_image_embeds.to(text_embeds.device).flatten(0, 1)

                teacher_attention_mask = attention_mask
                teacher_position_ids = torch.cumsum(teacher_attention_mask, dim=1) - 1
                teacher_position_ids[teacher_position_ids < 0] = 0

                outputs = teacher_language_model(
                    attention_mask=teacher_attention_mask,
                    position_ids=teacher_position_ids,
                    inputs_embeds=text_embeds,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]

                if und_image_embeds is not None and und_image_idx.any():
                    hidden_states[und_image_idx] = und_image_embeds.to(hidden_states.device).flatten(0, 1)
                if aux_image_embeds is not None and aux_image_idx.any():
                    hidden_states[aux_image_idx] = aux_image_embeds.to(hidden_states.device).flatten(0, 1)

                hidden_states = teacher_bundle["action_dit_connector"](hidden_states)
                hidden_states = teacher_bundle["action_dit_norm"](hidden_states)
                hidden_states = teacher_bundle["action_dit_projector"](hidden_states)

                token_counts = und_image_idx.sum(dim=1)
                if not torch.all(token_counts == token_counts[0]):
                    raise RuntimeError(f"Inconsistent und token counts for loc_repa teacher feature extraction: {token_counts.tolist()}")
                num_tokens = int(token_counts[0].item())
                batch_size = hidden_states.shape[0]
                hidden_dim = hidden_states.shape[-1]
                prefix_tokens = hidden_states[und_image_idx].reshape(batch_size, num_tokens, hidden_dim)
                return prefix_tokens

        combined_und_images = torch.cat([fps_image, map_image], dim=0)
        loc_task_id = torch.zeros_like(task_id)

        student_modules = [
            self.model.action_dit_connector,
            self.model.action_dit_norm,
            self.model.action_dit_projector,
        ]
        student_states = [
            (
                module,
                module.training,
                [param.requires_grad for param in module.parameters()],
            )
            for module in student_modules
        ]
        for module in student_modules:
            module.requires_grad_(False)
            module.eval()

        try:
            with torch.enable_grad():
                (
                    _,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                    _,
                    combined_img_idx,
                    combined_image_embeds,
                    _,
                ) = self.prepare_inputs_labels_for_multimodal(
                    loc_input_ids,
                    None,
                    attention_mask,
                    None,
                    loc_labels,
                    None,
                    combined_und_images,
                    None,
                    None,
                    None,
                    loc_task_id,
                )

                und_img_idx = combined_img_idx[:combined_img_idx.size(0)//2, ...]
                aux_img_idx = combined_img_idx[combined_img_idx.size(0)//2:, ...]
                und_image_embeds = combined_image_embeds[:combined_image_embeds.size(0)//2, ...]
                aux_image_embeds = combined_image_embeds[combined_image_embeds.size(0)//2:, ...]

                position_ids = torch.cumsum(attention_mask, dim=1) - 1
                position_ids[position_ids < 0] = 0

                outputs = self.model.language_model(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]

                if und_image_embeds is not None and und_img_idx is not None:
                    hidden_states[und_img_idx] = und_image_embeds.to(hidden_states.device).flatten(0, 1)
                if aux_image_embeds is not None and aux_img_idx is not None:
                    hidden_states[aux_img_idx] = aux_image_embeds.to(hidden_states.device).flatten(0, 1)

                hidden_states = self.get_model().action_dit_connector(hidden_states)
                hidden_states = self.get_model().action_dit_norm(hidden_states)
                hidden_states = self.get_model().action_dit_projector(hidden_states)

                token_counts = und_img_idx.sum(dim=1)
                if not torch.all(token_counts == token_counts[0]):
                    raise RuntimeError(f"Inconsistent und token counts for loc_repa feature extraction: {token_counts.tolist()}")
                num_tokens = int(token_counts[0].item())
                batch_size = hidden_states.shape[0]
                hidden_dim = hidden_states.shape[-1]
                prefix_tokens = hidden_states[und_img_idx].reshape(batch_size, num_tokens, hidden_dim)
                return prefix_tokens
        finally:
            for module, was_training, grad_flags in student_states:
                for param, grad_flag in zip(module.parameters(), grad_flags):
                    param.requires_grad_(grad_flag)
                module.train(was_training)

    def _compute_loc_repa_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        sigmas: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_type = getattr(self.config, "loc_repa_loss_type", "cosine")
        if loss_type != "cosine":
            raise ValueError(f"Unsupported loc_repa_loss_type: {loss_type}")

        student_feat = F.normalize(student_feat, dim=-1)
        teacher_feat = F.normalize(teacher_feat.detach(), dim=-1)
        token_loss = 1.0 - (student_feat * teacher_feat).sum(dim=-1)
        sample_loss = token_loss.mean(dim=-1)

        timestep_weight_type = getattr(self.config, "loc_repa_timestep_weight", "linear_1m_sigma")
        if timestep_weight_type == "linear_1m_sigma":
            weight = (1.0 - sigmas).clamp(min=0).squeeze()
        else:
            raise ValueError(f"Unsupported loc_repa_timestep_weight: {timestep_weight_type}")

        sample_loss = sample_loss * weight
        return (sample_loss * loss_mask[:, 1]).mean().to(torch.float32)

    def forward_for_loc_repa_loss(
        self,
        sigmas: torch.Tensor,
        pred_pixels_input: torch.Tensor,
        aux_loc_input_ids: torch.Tensor,
        aux_loc_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        und_image_map: torch.Tensor,
        gen_image: torch.Tensor,
        task_id: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.get_loc_repa_teacher() is None:
            raise RuntimeError("loc_repa teacher is not initialized before forward.")

        student_feat = self._extract_loc_repa_prefix_features(
            fps_image=pred_pixels_input,
            map_image=und_image_map,
            loc_input_ids=aux_loc_input_ids,
            loc_labels=aux_loc_labels,
            attention_mask=attention_mask,
            task_id=task_id,
            use_teacher=False,
        )
        teacher_feat = self._extract_loc_repa_prefix_features(
            fps_image=gen_image,
            map_image=und_image_map,
            loc_input_ids=aux_loc_input_ids,
            loc_labels=aux_loc_labels,
            attention_mask=attention_mask,
            task_id=task_id,
            use_teacher=True,
        )
        return self._compute_loc_repa_loss(student_feat, teacher_feat, sigmas, loss_mask)

    def _normalize_for_external_loc_model(self, image_01: torch.Tensor) -> torch.Tensor:
        target_size = int(getattr(self.config, "external_loc_input_size", 224))
        if image_01.shape[-1] != target_size or image_01.shape[-2] != target_size:
            image_01 = F.interpolate(
                image_01,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = torch.tensor([0.485, 0.456, 0.406], device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
        return (image_01 - mean) / std

    def _forward_external_loc_model(self, fps_image: torch.Tensor, map_image: torch.Tensor) -> torch.Tensor:
        external_model = self.get_external_localization_model()
        if external_model is None:
            raise RuntimeError("use_external_loc_model=True but external localization model is not attached.")
        external_model.eval()
        model_device = next(external_model.parameters()).device
        model_dtype = next(external_model.parameters()).dtype
        fps_image = fps_image.to(device=model_device, dtype=model_dtype)
        map_image = map_image.to(device=model_device, dtype=model_dtype)
        return external_model(fps_image, map_image)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)

    # ==========================================
    # 3. [NEW] Forward Implementation
    # ==========================================
    def forward(
        self,
        input_ids: torch.LongTensor = None, #torch.Size([128, 707])
        attention_mask: Optional[torch.Tensor] = None, #torch.Size([128, 707])
        position_ids: Optional[torch.LongTensor] = None, #None
        past_key_values: Optional[List[torch.FloatTensor]] = None, #None
        inputs_embeds: Optional[torch.FloatTensor] = None, #None
        labels: Optional[torch.LongTensor] = None, #torch.Size([128, 707])
        ids: Optional[list] = None, #len=128 * file_frame
        i_s_pos: Optional[list] = None, #None
        use_cache: Optional[bool] = None, #None
        output_attentions: Optional[bool] = None, #output_attentions
        output_hidden_states: Optional[bool] = None, #None

        # Custom Inputs from Dataset
        gen_image: Optional[torch.FloatTensor] = None, #torch.Size([128, 3, 448, 448])
        und_image: Optional[torch.FloatTensor] = None, #torch.Size([128, 3, 448, 448])
        aux_image: Optional[torch.FloatTensor] = None, # [NEW] Support Aux Image (Wrist/Map) #torch.Size([128, 3, 448, 448])
        loc_fps_image: Optional[torch.FloatTensor] = None, # [NEW] External loc model fps input [BS,3,224,224]
        loc_map_image: Optional[torch.FloatTensor] = None, # [NEW] External loc model map input [BS,3,224,224]
        actions: Optional[torch.FloatTensor] = None,   # [NEW] GT 5D Pose [BS, 1, 5]  # torch.Size([128, 5])
        loss_mask: Optional[torch.FloatTensor] = None, # [NEW] [BS, 2] -> [Loc_Weight, Gen_Weight]  # torch.Size([128, 2])
        aux_loc_input_ids: torch.LongTensor = None,
        aux_loc_labels: Optional[torch.LongTensor] = None,
        aux_loc_attention_mask: Optional[torch.Tensor] = None,

        # Others
        grid_thw: Optional[torch.FloatTensor] = None, # None
        image_sizes: Optional[List[List[int]]] = None, # None
        return_dict: Optional[bool] = None, # None
        task_id: Optional[torch.FloatTensor] = None, # [NEW] [BS] -> 0: Loc, 1: Gen #torch.Size([128])
        **kwargs #dict_keys(['map_id', 'raw_prompt', 'map_name', 'pose_dict', 'num_items_in_batch']) #'num_items_in_batch': tensor(18631, device='cuda:0')
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # bs=8显存占用=5896MiB
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loc_indices = (task_id == 0).nonzero(as_tuple=True)[0]
        gen_indices = (task_id == 1).nonzero(as_tuple=True)[0]
        if getattr(self.config, "use_pi05_action_dit", False):
            actions = torch.cat([
                actions,
                torch.full((actions.size(0), 1, self.get_model()._default_pi05_action_dim - actions.size(-1)), 0.0, dtype=torch.bfloat16).to(actions.device)
            ], dim=-1)

        # --- A. Input Preparation ---
        # Note: We merge und_image and aux_image logic.
        # In Dataset: Localization task has `und_image`(FPS) and `aux_image`(Map).
        # In Dataset: Generation task has `und_image`(Map) and `aux_image`(Empty).
        # We need to concat them for processing if both exist.

        if ( (not getattr(self.config, "use_vit_cls_regression_head", False)) and (not getattr(self.config, "use_codex_vit_regression_head", False)) ) \
            or (getattr(self.config, "use_vit_cls_regression_head", False) and loss_mask[gen_indices][:, 1].sum() > 0) \
            or (getattr(self.config, "use_codex_vit_regression_head", False) and loss_mask[gen_indices][:, 1].sum() > 0):
            combined_und_images = und_image
            if aux_image is not None:# and aux_image.sum() != 0:
                if combined_und_images is not None:
                    # Assuming simple concat in batch dimension for processing, then splitting?
                    # Or Dataset handles embedding replacement.
                    # Given prepare_inputs logic, we rely on input_ids tokens.
                    # If there are two <image> tokens in Loc Prompt, prepare_inputs handles replacement sequentially.
                    # Just need to ensure `und_images` passed to prepare_inputs contains all pixel values.
                    combined_und_images = torch.cat([und_image, aux_image], dim=0) # Careful with indexing mapping!
                    # Actually, simpler: The dataset currently provides und_image and aux_image separately.
                    # But input_ids has multiple <image> placeholders.
                    # For simplicity in this adaptation, let's assume `und_image` arg passed to `prepare_inputs`
                    # should contain ALL images referenced by input_ids (except gen target).
                    # Your Dataset logic: und_image is [1, C, H, W], aux_image is [1, C, H, W].
                    # If task=Loc, we have 2 images. We should stack them.

                    # Check Batch Size:
                    # If input_ids is [BS, Seq], und_image is [BS, 1, C, H, W] (from dataset collator likely [BS, 1, C, H, W])
                    pass

            # For strict compatibility, let's pass `und_image` (which might be a stack of images if collated correctly).
            # Assuming the standard collator stacks images into [Total_Images_In_Batch, C, H, W].

            if inputs_embeds is None:
                # with torch.no_grad():
                if 1:
                    ( # return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds, combined_img_idx, combined_image_embeds, bidr_attention_mask
                        _,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        inputs_embeds,
                        labels,
                        target_image_embeds, #latents
                        combined_img_idx,
                        combined_image_embeds,
                        bidr_attention_mask
                    ) = self.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        labels,
                        gen_image[gen_indices],
                        combined_und_images, # Pass und_image (which contains all input visuals)
                        grid_thw,
                        i_s_pos,
                        image_sizes,
                        task_id,
                    )
                    und_img_idx = combined_img_idx[:combined_img_idx.size(0)//2, ...] #und_img_idx,sum()=32768 #32768/256=128.0
                    aux_img_idx = combined_img_idx[combined_img_idx.size(0)//2:, ...]#aux_img_idx.sum()=tensor(14592, device='cuda:0') #14592/256=57
                    und_image_embeds = combined_image_embeds[:combined_image_embeds.size(0)//2, ...]#torch.Size([128, 256, 896])
                    aux_image_embeds = combined_image_embeds[combined_image_embeds.size(0)//2:, ...]#torch.Size([128, 256, 896])

            if (not getattr(self.config, "use_vit_regression_head", False)) \
                or (getattr(self.config, "use_vit_regression_head", False) and loss_mask[gen_indices][:, 1].sum() > 0):
            # --- B. Main LLM Forward (Understanding) ---
                # with torch.no_grad():
                if 1:
                    position_ids = torch.cumsum(attention_mask, dim=1) - 1
                    position_ids[position_ids < 0] = 0
                    # bs=8显存占用=6186MiB
                    # inputs_embeds=torch.Size([16, 515, 896])

                    outputs = self.model.language_model(
                        attention_mask=attention_mask, #torch.Size([128, 707])
                        position_ids=position_ids, #torch.Size([128, 707])
                        inputs_embeds=inputs_embeds, #torch.Size([128, 707, 896])
                        output_hidden_states=True,
                        return_dict=return_dict, #True
                        use_cache=False
                    )

                    # Last Hidden State from LLM: [BS, Seq_Len, Hidden_Size]
                    # This contains contextualized features of both text and images.
                    hidden_states = outputs.hidden_states[-1] #torch.Size([128, 707, 896])
                    # bs=8显存占用=6654MiB
                    # # 在 --- B. Main LLM Forward --- 之后插入
                    # logging.info(f"DEBUG Check:")
                    # logging.info(f"  Input IDs Shape: {input_ids. shape}")
                    # logging.info(f"  Hidden States Shape: {hidden_states.shape}")
                    # logging.info(f"  Combined Img Idx Shape: {combined_img_idx.shape}") # 关键！看是不是 2*BS
                    # logging.info(f"  Valid Lens Sample (0-5): {attention_mask.sum(dim=1)[:5]}")
                    # logging.info(f"  Loss Mask Sum (Loc/Gen): {loss_mask.sum(dim=0)}")

                    # Re-fill und_image embeddings (Skip Connection logic from UniLIP)
                    if und_image_embeds is not None and und_img_idx is not None:
                        hidden_states[und_img_idx] = und_image_embeds.to(hidden_states.device).flatten(0,1)

                    is_loc_task = (task_id == 0)#is_loc_task.shape=torch.Size([128])#is_loc_task.sum()=tensor(57, device='cuda:0')
                    if aux_image_embeds is not None and und_img_idx is not None: #aux_img_idx.sum()/128 = 57
                        hidden_states[aux_img_idx] = aux_image_embeds[is_loc_task].to(hidden_states.device).flatten(0,1)#hidden_states[aux_img_idx].shape=torch.Size([14592, 896]) #aux_image_embeds[is_loc_task].shape=torch.Size([57, 256, 896])

        # --- C. Task Branching based on Loss Mask ---
        # loss_mask: [BS, 2] -> [Loc, Gen]
        if loss_mask is None:
            # loss_mask = torch.ones(hidden_states.shape[0], 2).to(hidden_states.device) # Default all on
            total_loss = torch.nn.MSELoss()(hidden_states, torch.clone(hidden_states.detach()))
        else:
            # ==========================================
            # 按索引拆分生成/定位样本
            # ==========================================
            # logging.info(f"loc_indices: {len(loc_indices)}, gen_indices: {len(gen_indices)}")

            # ==========================================
            # Branch 1: GENERATION (DiT Path)
            # ==========================================
            if loss_mask[gen_indices][:, 1].sum() > 0: # If any sample needs Generation
                is_gen_task = (task_id == 1)#is_gen_task.sum()=tensor(71, device='cuda:0')
                # TODO_Done 如果这里使用[is_gen_task]过滤后再进行llm_connector和dit，能节约显存，但是由于[is_gen_task]的数目不一定恰好等于2的次方，所以可能会影响cuda加速运算。除非数据集collactor手动设置loc:gen=64:64。
                genbrh_bidr_attention_mask = bidr_attention_mask[gen_indices]
                genbrh_position_ids = position_ids[gen_indices]
                genbrh_hidden_states = hidden_states[gen_indices]
                genbrh_target_image_embeds = target_image_embeds#[gen_indices]
                genbrh_attention_mask = attention_mask[gen_indices]
                genbrh_loss_mask = loss_mask[gen_indices]
                genbrh_aux_loc_input_ids = aux_loc_input_ids[gen_indices]
                genbrh_aux_loc_labels = aux_loc_labels[gen_indices]
                genbrh_aux_loc_attention_mask = aux_loc_attention_mask[gen_indices]
                genbrh_und_image = und_image[gen_indices]
                genbrh_gen_image = gen_image[gen_indices]
                genbrh_loc_map_image = loc_map_image[gen_indices] if loc_map_image is not None else None
                genbrh_task_id = task_id[gen_indices]
                genbrh_actions = actions[gen_indices]

                # 1. Process features via LLM Connector
                img_hidden_states = self.model.llm_connector(
                    attention_mask=genbrh_bidr_attention_mask,#torch.Size([128, 1, 707, 707])
                    position_ids=genbrh_position_ids,#torch.Size([128, 707])
                    inputs_embeds=genbrh_hidden_states,#torch.Size([128, 707, 896])
                    output_hidden_states=True,
                    return_dict=return_dict,#True
                    use_cache=False
                ).hidden_states[-1]

                # 2. Project to DiT Caption Channel
                img_hidden_states = self.get_model().projector(img_hidden_states) #torch.Size([128, 707, 2304])
                # bs=8显存占用=6884MiB
                # 3. Calculate DiT Loss
                if genbrh_target_image_embeds is not None:#target_image_embeds=torch.Size([128, 32, 16, 16])
                    latents = genbrh_target_image_embeds # [BS_Gen, C, H, W]
                    bsz = latents.shape[0] #128
                    noise = torch.randn_like(latents, device=latents.device)#torch.Size([128, 32, 16, 16])
                    u = compute_density_for_timestep_sampling(weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0) #torch.Size([128])
                    indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long() #torch.Size([128])
                    timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)#torch.Size([128])
                    sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)#torch.Size([128, 1, 1, 1])

                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise #torch.Size([128, 32, 16, 16])

                    # Forward DiT
                    noise_pred, _ = self._forward_dit_with_repa_hidden(
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        encoder_hidden_states=img_hidden_states,
                        encoder_attention_mask=genbrh_attention_mask,
                        detach_condition=False,
                    )

                    target = noise - latents #torch.Size([128, 32, 16, 16])
                    # Compute raw MSE loss per sample [BS, ...]
                    gen_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")#torch.Size([128, 32, 16, 16])
                    gen_loss = gen_loss.mean(dim=[1, 2, 3]) # [BS]

                    # Apply Mask: Only count loss for Gen samples
                    # masked_gen_loss = (gen_loss * loss_mask[:, 1]).mean()#loss_mask[:, 1].sum()=tensor(71., device='cuda:0', dtype=torch.bfloat16)
                    masked_gen_loss = (gen_loss * genbrh_loss_mask[:, 1]).mean()
                    masked_repa_loss = torch.zeros((), device=masked_gen_loss.device, dtype=torch.float32)
                    masked_loc_repa_loss = torch.zeros((), device=masked_gen_loss.device, dtype=torch.float32)
                    masked_loc_aux_loss = torch.zeros((), device=masked_gen_loss.device, dtype=torch.float32)
                    # bs=8显存占用=6884MiB
                    pred_pixels_input = None
                    if getattr(self.config, "is_loc_aux_loss", False) or getattr(self.config, "is_loc_repa_loss", False):
                        pred_pixels_input, _ = self._decode_pred_pixels_input_from_flow(
                            sigmas=sigmas,
                            noisy_latents=noisy_latents,
                            noise_pred=noise_pred,
                            und_image_map=genbrh_und_image,
                        )

                    if getattr(self.config, "is_loc_repa_loss", False) and pred_pixels_input is not None:
                        masked_loc_repa_loss = self.forward_for_loc_repa_loss(
                            sigmas=sigmas,
                            pred_pixels_input=pred_pixels_input,
                            aux_loc_input_ids=genbrh_aux_loc_input_ids,
                            aux_loc_labels=genbrh_aux_loc_labels,
                            attention_mask=genbrh_aux_loc_attention_mask,
                            und_image_map=genbrh_und_image,
                            gen_image=genbrh_gen_image,
                            task_id=torch.zeros_like(genbrh_task_id),
                            loss_mask=genbrh_loss_mask,
                        )

                    if getattr(self.config, "is_repa_loss", False):
                        masked_repa_loss = self.forward_for_repa_loss(
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            encoder_hidden_states=img_hidden_states,
                            encoder_attention_mask=genbrh_attention_mask,
                            clean_image=genbrh_gen_image,
                            loss_mask=genbrh_loss_mask,
                        )

                    # =========================================================
                    # [NEW] Auxiliary Localization Loss (Consistency Check)
                    # =========================================================
                    # 仅当训练生成任务，且 actions (GT Pose) 存在时计算
                    # 并且为了显存安全，可能只对部分样本计算，或者需要 gradient checkpointing
                    if getattr(self.config, 'is_loc_aux_loss', False) and genbrh_actions is not None and pred_pixels_input is not None:
                        # actions [BS, 1, 5] 即使是 Gen 任务，Dataset 也应该把 pose 传进来
                        masked_loc_aux_loss = self.forward_for_aux_loc_loss(
                            sigmas, #torch.Size([16, 1, 1, 1])
                            noisy_latents, #torch.Size([16, 32, 16, 16])
                            noise_pred, #torch.Size([16, 32, 16, 16])
                            genbrh_aux_loc_input_ids, #torch.Size([16, 617])
                            genbrh_aux_loc_labels, #torch.Size([16, 617])
                            genbrh_aux_loc_attention_mask, #torch.Size([128, 707])
                            genbrh_und_image, #torch.Size([16, 3, 448, 448])
                            genbrh_gen_image, #torch.Size([16, 3, 448, 448])
                            genbrh_loc_map_image,
                            grid_thw, #None
                            i_s_pos, #None
                            image_sizes, #None
                            torch.zeros_like(genbrh_task_id), #torch.Size([16])
                            return_dict, #True
                            genbrh_actions, #torch.Size([16, 1, 5])
                            genbrh_loss_mask, #torch.Size([16, 2])
                            pred_pixels_input=pred_pixels_input,
                        )

            else:
                masked_gen_loss = torch.nn.MSELoss()(actions, torch.clone(actions.detach())).to(torch.float32)
                masked_repa_loss = torch.nn.MSELoss()(actions, torch.clone(actions.detach())).to(torch.float32)
                masked_loc_aux_loss = torch.nn.MSELoss()(actions, torch.clone(actions.detach())).to(torch.float32)
                masked_loc_repa_loss = torch.nn.MSELoss()(actions, torch.clone(actions.detach())).to(torch.float32)
            # bs=8显存占用=13316MiB
            # ==========================================
            # Branch 2: LOCALIZATION (Flow Matching Path)
            # ==========================================
            if loss_mask[loc_indices][:, 0].sum() > 0: # If any sample needs Localization
                if getattr(self.config, "use_external_loc_model", False):
                    locbrh_actions = actions[loc_indices]
                    locbrh_fps_image = loc_fps_image[loc_indices] if loc_fps_image is not None else und_image[loc_indices]
                    locbrh_map_image = loc_map_image[loc_indices] if loc_map_image is not None else aux_image[loc_indices]
                    locbrh_actions_pred = self._forward_external_loc_model(locbrh_fps_image, locbrh_map_image)
                    if getattr(self.config, "external_loc_use_circular_loss", True):
                        masked_loc_loss = self._compute_codex_loc_regression_loss(locbrh_actions_pred, locbrh_actions).to(torch.float32)
                    else:
                        masked_loc_loss = torch.nn.MSELoss()(locbrh_actions_pred, locbrh_actions).to(torch.float32)
                    masked_loc_loss_valid5 = masked_loc_loss.detach().to(torch.float32)

                elif getattr(self.config, "use_codex_vit_regression_head", False):
                    locbrh_actions = actions[loc_indices]#torch.Size([128, 5])
                    locbrh_und_image = und_image[loc_indices]
                    locbrh_aux_image = aux_image[loc_indices]
                    locbrh_actions_pred = self._forward_codex_vit_regression_head(
                        locbrh_und_image,
                        locbrh_aux_image,
                    )
                    if getattr(self.config, "loc_use_circular_loss", True):
                        masked_loc_loss = self._compute_codex_loc_regression_loss(locbrh_actions_pred, locbrh_actions).to(torch.float32)
                    else:
                        masked_loc_loss = torch.nn.MSELoss()(locbrh_actions_pred, locbrh_actions).to(torch.float32)
                    masked_loc_loss_valid5 = masked_loc_loss.detach().to(torch.float32)

                elif getattr(self.config, "use_vit_cls_regression_head", False):
                    locbrh_actions = actions[loc_indices]#torch.Size([128, 5])
                    locbrh_und_image = und_image[loc_indices]
                    locbrh_aux_image = aux_image[loc_indices]
                    locbrh_actions_pred = self._forward_vit_regression_head(
                        locbrh_und_image,
                        locbrh_aux_image,
                    )
                    if getattr(self.config, "loc_use_circular_loss", True):
                        masked_loc_loss = self._compute_codex_loc_regression_loss(locbrh_actions_pred, locbrh_actions).to(torch.float32)
                    else:
                        masked_loc_loss = torch.nn.MSELoss()(locbrh_actions_pred, locbrh_actions).to(torch.float32)
                    masked_loc_loss_valid5 = masked_loc_loss.detach().to(torch.float32)

                elif getattr(self.config, "use_vit_regression_head", False):
                    locbrh_actions = actions[loc_indices]#torch.Size([128, 5])
                    locbrh_und_image_embeds = und_image_embeds[loc_indices]
                    locbrh_aux_image_embeds = aux_image_embeds[loc_indices]

                    # locbrh_und_feature = self.get_model().img_pooler(locbrh_und_image_embeds)
                    # locbrh_aux_feature = self.get_model().img_pooler(locbrh_aux_image_embeds)
                    # print(locbrh_und_image_embeds.shape)
                    # locbrh_und_feature = locbrh_und_image_embeds[:, 0, :]
                    # locbrh_aux_feature = locbrh_aux_image_embeds[:, 0, :]
                    # locbrh_concat_feature = torch.cat([locbrh_und_feature, locbrh_aux_feature], dim=-1)
                    locbrh_concat_feature = self.get_model().cross_view_fusion(locbrh_und_image_embeds, locbrh_aux_image_embeds)
                    locbrh_concat_feature = self.get_model().action_dit_norm(locbrh_concat_feature)
                    if getattr(self.config, "is_action_dit_projector", False):
                        locbrh_concat_feature = self.get_model().action_dit_projector(locbrh_concat_feature)
                    locbrh_actions_pred = self.get_model().regression_loc_head(locbrh_concat_feature)
                    if getattr(self.config, "loc_use_circular_loss", True):
                        masked_loc_loss = self._compute_codex_loc_regression_loss(locbrh_actions_pred, locbrh_actions).to(torch.float32)
                    else:
                        masked_loc_loss = torch.nn.MSELoss()(locbrh_actions_pred, locbrh_actions).to(torch.float32)
                    masked_loc_loss_valid5 = masked_loc_loss.detach().to(torch.float32)

                else:
                    # actions: [BS, 1, 5]
                    locbrh_actions = actions[loc_indices]#torch.Size([128, 5])
                    locbrh_hidden_states = hidden_states[loc_indices]
                    locbrh_attention_mask = attention_mask[loc_indices]
                    locbrh_position_ids = position_ids[loc_indices]
                    locbrh_loss_mask = loss_mask[loc_indices]
                    # # TODO 这里也一样！如果使用[is_loc_task]过滤actions和其他中间tensor后再进行embed_action_suffix和action_dit，能节约显存，但是由于[is_loc_task]的数目不一定恰好等于2的次方，所以可能会影响cuda加速运算。除非数据集collactor手动设置loc:gen=64:64。
                    # 1. Flow Matching Setup
                    # Sample Noise & Time
                    noise = self.sample_noise(locbrh_actions.shape, locbrh_actions.device)#torch.Size([128, 5])
                    time = self.sample_time(locbrh_actions.shape[0], locbrh_actions.device)#torch.Size([128])

                    # Interpolate: x_t = t * noise + (1-t) * x_1 (Actions)
                    time_expanded = time[:, None, None].to(locbrh_actions.dtype) # [BS, 1, 1] #torch.Size([128, 1, 1])
                    x_t = time_expanded * noise + (1 - time_expanded) * locbrh_actions # 带噪声的中间向量x_t #torch.Size([128, 1, 5])

                    u_t = noise - locbrh_actions # 需要模型预测的 Velocity Target 速度场 #torch.Size([128, 5])

                    # 2. Prepare Inputs for Action Connector
                    # We treat LLM hidden_states as "Prefix" (Context)
                    # Embed the suffix (Noisy Action + Time)
                    suffix_emb, adarms_cond = self.embed_action_suffix(
                        x_t, #torch.Size([128, 1, 5])
                        time, #torch.Size([128])
                        llm_hidden_size=1024 if getattr(self.config, "use_pi05_action_dit", False) else self.model.config.text_config.hidden_size,
                        device=locbrh_actions.device,
                        dtype=locbrh_hidden_states.dtype
                    ) # suffix_emb: [BS, 1, Hidden] #torch.Size([128, 1, 896])

                    if getattr(self.config, "use_pi05_action_dit", False):
                        locbrh_hidden_states = self.get_model().action_dit_connector(locbrh_hidden_states)
                    # 防止；anguage_model输出的last_hidden_state出现max=266，min=-256，而导致梯度nan
                    locbrh_hidden_states = self.get_model().action_dit_norm(locbrh_hidden_states)
                    # action_emb = self.get_model().action_norm(action_emb)

                    # scaler = self.model.config.text_config.hidden_size ** 0.5
                    # suffix_emb = suffix_emb * scaler

                    # 3. Concatenate & Forward Action Connector
                    # Context (LLM Output) + Suffix (Action)
                    # We need to construct attention mask so Suffix sees Context, but standard Causal mask is fine usually

                    # Concat Embeddings
                    # Scatter 填充 # “右移填空” (Right Shift & Fill) 操作 # 实现"Left Padding" 或 "Packing"的效果
                    bs, seq_len, hidden_dim = locbrh_hidden_states.shape
                    valid_lens = locbrh_attention_mask.sum(dim=1).long()

                    # action_dit_inputs = torch.cat([hidden_states, suffix_emb], dim=1) # [BS, Seq+1, Hidden] #hidden_states=torch.Size([128, 707, 896]) #suffix_emb=torch.Size([128, 1, 896]) #action_dit_inputs #torch.Size([128, 708, 896])
                    action_dit_inputs = torch.cat([locbrh_hidden_states, torch.zeros_like(suffix_emb)], dim=1)
                    target_indices = valid_lens.view(-1, 1, 1).expand(-1, 1, hidden_dim)#把 suffix_emb 放到 valid_lens 的位置 # 构造索引：我们需要修改的位置是 (b, valid_lens[b]) # view(-1, 1, 1) 是为了广播到 hidden_dim
                    action_dit_inputs = action_dit_inputs.scatter(1, target_indices, suffix_emb)

                    # Extend Masks
                    # 1 for Action token (visible)
                    # action_mask = torch.ones((bsz, 1, 1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                    # action_mask = torch.ones((bsz, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                    # action_dit_att_mask = torch.cat([attention_mask, action_mask], dim=1) #torch.Size([128, 708])
                    action_dit_att_mask = torch.cat([locbrh_attention_mask, torch.zeros((bs, 1), device=locbrh_attention_mask.device, dtype=locbrh_attention_mask.dtype)], dim=1)
                    mask_indices = valid_lens.view(-1, 1)
                    action_dit_att_mask = action_dit_att_mask.scatter(1, mask_indices, 1)

                    # Position IDs
                    # action_pos_id = position_ids.max(dim=1)[0].unsqueeze(1) + 1
                    # action_dit_pos_ids = torch.cat([position_ids, action_pos_id], dim=1) #position_ids=torch.Size([128, 707]) #action_pos_id=torch.Size([128, 1])
                    action_dit_pos_ids = torch.cat([locbrh_position_ids, torch.zeros((bs, 1), device=locbrh_position_ids.device, dtype=locbrh_position_ids.dtype)], dim=1)
                    # Action 的 Pos ID 应该是上一个 token 的 pos + 1，或者直接就是 valid_lens (如果从0开始)
                    # 假设你的 position_ids 在 padding 处是 0 或其他，我们这里显式计算一下 action 的 pos
                    action_pos_ids = valid_lens.view(-1, 1) # Action 的位置索引就是它的序列位置
                    # action_dit_pos_ids = action_dit_pos_ids.scatter(1, mask_indices, action_pos_ids)
                    # [核心修改] 生成掩码并填充
                    # 我们要找到所有 index >= valid_lens 的位置
                    # 构造一个 range 矩阵: [0, 1, 2, ..., Seq]
                    current_seq_len = action_dit_pos_ids.shape[1]
                    range_ids = torch.arange(current_seq_len, device=locbrh_position_ids.device).unsqueeze(0) # [1, Seq+1]
                    # 生成掩码：如果当前位置 index >= valid_lens，则为 True
                    # [BS, 1] vs [1, Seq+1] -> Broadcast -> [BS, Seq+1]
                    mask_after_valid = range_ids >= valid_lens.view(-1, 1)
                    # 使用 torch.where 进行批量填充
                    # 逻辑：Mask 为 True 的地方填入 action_pos_id，False 的地方保持原样
                    action_dit_pos_ids = torch.where(
                        mask_after_valid,
                        action_pos_ids,      # 广播填充 [BS, 1] -> [BS, Mask区域]
                        action_dit_pos_ids  # 保持原值
                    )
                    # 这里action_dit_inputs和action_dit_pos_ids有一个风险点，为了将suffix_emb和action_pos_ids拼接到正确的位置，我们先使用zeros填充到正确的seq长度，然后再使用scatter找到正确位置valid_lens填充。而这样做会导致原先padding位置的tensor被zeros替代，虽然action_dit_att_mask不受影响且会忽略padding位置的tensor的梯度计算，但还是有风险(剧烈数值波动、某些bf16计算、特定FlashAttention算子等)。
                    # 还有一个更糟糕的情况，如果手动将attention_mask的padding位置的position_id正确地替代到action_dit_pos_ids上是可以实现的。但是将hidden_states在padding位置的embed替代到正确的action_dit_inputs的位置是不可能的，因为hidden_states每个位置的embed是由所有位置的embed计算得到的。
                    # 综上所述，已经修改action_dit的代码来适配adarms_cond。且使用zeros+scatter(valid_lens)填充hidden_states和action_embeds。

                    if getattr(self.config, "is_action_dit_projector", False):
                        action_dit_inputs = self.get_model().action_dit_projector(action_dit_inputs)

                    if getattr(self.config, "use_pi05_action_dit", False):
                        suffix_output = self.get_model().action_dit(
                            inputs_embeds=action_dit_inputs,
                            attention_mask=self._prepare_attention_masks_4d_from_attn_masks_1d(action_dit_att_mask),
                            position_ids=action_dit_pos_ids,
                            use_cache=False,
                            adarms_cond=adarms_cond,
                        )
                        action_hidden = suffix_output.last_hidden_state
                    else:
                        # bs=8显存占用=13316MiB
                        # Forward Action Connector (InternVL Slice)
                        # Reuse bidr mask logic or standard causal. Since it's InternVL, it expects eager/causal usually.
                        # For simplicity, we assume bidr mask logic handles the sequence extension as default.
                        # 注意在OpenPi0.5中使用的gemma_expert_model还会接受adarms_cond(一个跟timestep有关的embedding)作为输入
                        if getattr(self.config, 'is_action_dit_dense_timestep', False):
                            if getattr(self.config, "is_loc_learnable_query", False):
                                action_outputs = self.action_dit_forward_with_adarmscond(
                                    hidden_states=self.loc_learnable_query,
                                    encoder_hidden_states=locbrh_hidden_states, #torch.Size([128, 708, 896])
                                    encoder_attention_mask=locbrh_attention_mask, #torch.Size([128, 708])
                                    encoder_position_ids=locbrh_position_ids, #torch.Size([128, 708])
                                    adarms_cond=adarms_cond,
                                )
                            else:
                                action_outputs = self.action_dit_forward_with_adarmscond(
                                    hidden_states=action_dit_inputs, #torch.Size([128, 708, 896])
                                    attention_mask=action_dit_att_mask, #torch.Size([128, 708])
                                    position_ids=action_dit_pos_ids, #torch.Size([128, 708])
                                    adarms_cond=adarms_cond,
                                    # 和Pi05的不同：除了没有使用adarms_cond之外，action_dit也没有使用full_att_2d_masks_4d
                                        # Pi05的language_model和DiT都使用了prefix双向，suffix单向的mask；
                                        # 而UniLIP的language_model和DiT使用了单向mask，但是中间的llm_connector使用了单向mask；
                                )
                            action_hidden = action_outputs
                        else:
                            if getattr(self.config, "is_loc_learnable_query", False):
                                action_outputs = self.model.action_dit(
                                    inputs_embeds=self.loc_learnable_query,
                                    encoder_inputs_embeds=locbrh_hidden_states, #torch.Size([128, 708, 896])
                                    encoder_attention_mask=locbrh_attention_mask, #torch.Size([128, 708])
                                    encoder_position_ids=locbrh_position_ids, #torch.Size([128, 708])
                                    output_hidden_states=True,
                                    return_dict=return_dict,
                                    use_cache=False
                                )
                            else:
                                action_outputs = self.model.action_dit(
                                    inputs_embeds=action_dit_inputs, #torch.Size([128, 708, 896])
                                    attention_mask=action_dit_att_mask, #torch.Size([128, 708])
                                    position_ids=action_dit_pos_ids, #torch.Size([128, 708])
                                    output_hidden_states=True,
                                    # adarms_cond=[None, adarms_cond],
                                    # 和Pi05的不同：除了没有使用adarms_cond之外，action_dit也没有使用full_att_2d_masks_4d
                                        # Pi05的language_model和DiT都使用了prefix双向，suffix单向的mask；
                                        # 而UniLIP的language_model和DiT使用了单向mask，但是中间的llm_connector使用了单向mask；
                                    return_dict=return_dict,
                                    use_cache=False
                                )

                            # Get output corresponding to the Action Token (Last token)
                            # output: [BS, Seq+1, Hidden]
                            action_hidden = action_outputs.hidden_states[-1] # [BS, seq+1, Hidden] # [:, -1:, :] # [BS, 1, Hidden] #torch.Size([128, 1, 896])

                    #### TODO
                    # 修改整个action_dit，适配q=action，kv=hidden_states的SANATransformer格式
                    # action_hidden = self.model.action_dit(
                    #     suffix_emb, #torch.Size([128, 32, 16, 16])
                    #     timestep=time, #128
                    #     encoder_hidden_states=hidden_states, # [BS, Seq, C] ##torch.Size([128, 707, 2304])
                    #     encoder_attention_mask=attention_mask, #torch.Size([128, 707])
                    #     return_dict=False
                    # )[0] #[BS, 1, Hidden]

                    if getattr(self.config, "is_loc_learnable_query", False):
                        action_hidden = action_hidden[:,-1,:]
                    else:
                        # Gather from the same indices we scattered to
                        gather_indices = valid_lens.view(-1,1,1).expand(-1, -1, hidden_dim)
                        action_hidden = action_hidden.gather(1, gather_indices) # [BS, 1, Hidden]
                    # bs=8显存占用=13316MiB
                    # 4. Final Projection (Velocity Prediction)
                    v_t_pred = self.get_model().action_out_proj(action_hidden) # [BS, 1, 5] #torch.Size([128, 1, 5])
                    # # 在 --- Localiztion Branch --- 内部，v_t_pred 计算出来后插入
                    # logging.info(f"  Action Pred Mean: {v_t_pred.mean().item():.4f}, Std: {v_t_pred.std().item():.4f}")
                    # logging.info(f"  Action GT Mean: {u_t.mean().item():.4f}, Std: {u_t.std().item():.4f}")

                    # 5. Calculate Loss (MSE)
                    loc_loss_full = F.mse_loss(v_t_pred.float(), u_t.float(), reduction="none") #torch.Size([128, 1, 5])
                    loc_loss = loc_loss_full.mean(dim=[1, 2]) # [BS] #torch.Size([128])

                    # Apply Mask: Only count loss for Loc samples
                    masked_loc_loss = (loc_loss * locbrh_loss_mask[:, 0]).mean()#loss_mask[:, 0].sum()=tensor(57., device='cuda:0', dtype=torch.bfloat16)

                    # monitor only: valid 5-dim velocity mse, no change to training objective
                    if getattr(self.config, "use_pi05_action_dit", False):
                        loc_loss_valid5 = loc_loss_full[..., :self.model.config.action_dim].mean(dim=[1, 2])  # [BS]
                        masked_loc_loss_valid5 = (loc_loss_valid5 * locbrh_loss_mask[:, 0]).mean().to(torch.float32)
                    else:
                        # keep same scale for non-pi05 path so logging code can stay uniform
                        masked_loc_loss_valid5 = masked_loc_loss.detach().to(torch.float32)
            else:
                masked_loc_loss = torch.nn.MSELoss()(hidden_states, torch.clone(hidden_states.detach())).to(torch.float32)
                masked_loc_loss_valid5 = masked_loc_loss.detach().to(torch.float32)

        alpha_loc_aux_loss = torch.tensor(self.model.config.alpha_loc_aux_loss).to(torch.float32)
        alpha_repa_loss = torch.tensor(getattr(self.model.config, "alpha_repa_loss", 0.0)).to(torch.float32)
        alpha_loc_repa_loss = torch.tensor(getattr(self.model.config, "alpha_loc_repa_loss", 0.0)).to(torch.float32)
        alpha_loc_loss = torch.tensor(self.model.config.alpha_loc_loss).to(torch.float32)
        total_loss = (
            masked_gen_loss
            + masked_repa_loss * alpha_repa_loss
            + masked_loc_loss * alpha_loc_loss
            + masked_loc_repa_loss * alpha_loc_repa_loss
            + masked_loc_aux_loss * alpha_loc_aux_loss
        )
        # logging.info(f"total_loss: {total_loss.detach().cpu().numpy().item():6f}, masked_loc_loss: {masked_loc_loss.detach().cpu().numpy().item():6f}, alpha_loc_loss: {alpha_loc_loss.detach().cpu().numpy().item():6f}, masked_gen_loss: {masked_gen_loss.detach().cpu().numpy().item():6f}, masked_loc_aux_loss: {masked_loc_aux_loss.detach().cpu().numpy().item():6f}, alpha_loc_aux_loss: {alpha_loc_aux_loss.detach().cpu().numpy().item():6f}")
        # 224 + lora=64
        # bs=8显存占用=13316MiB
        # bs=16显存占用=20846MiB
        # bs=32显存占用=35678MiB
        #
        CausalLMOutputs = CausalLMOutputWithPast(
            loss=total_loss,
            logits=None, # Not used
            past_key_values=None, #outputs.past_key_values if outputs is not None else None,
            hidden_states=None, #outputs.hidden_states if outputs is not None else None,
            attentions=None, #outputs.attentions if outputs is not None else None,
        )

        CausalLMOutputs.extras = {
            "other_info": {
                "loc_indices": len(loc_indices),
                "gen_indices": len(gen_indices),
                "total_loss": total_loss.detach().cpu().numpy().item(),
                "loc_loss_valid5": masked_loc_loss_valid5.detach().cpu().numpy().item(),
                "alpha_weighted_loc_loss_valid5": (masked_loc_loss_valid5 * alpha_loc_loss).detach().cpu().numpy().item(),
                "alpha_weighted_loc_loss_valid5_over_gen_loss": (
                    (masked_loc_loss_valid5 * alpha_loc_loss / masked_gen_loss).detach().cpu().numpy().item()
                    if masked_gen_loss.item() > 0 else None
                ),
                "loc_loss": masked_loc_loss.detach().cpu().numpy().item(),
                "alpha_loc": alpha_loc_loss.detach().cpu().numpy().item(),
                "gen_loss": masked_gen_loss.detach().cpu().numpy().item(),
                "alpha_weighted_loc_loss": (masked_loc_loss * alpha_loc_loss).detach().cpu().numpy().item(),
                "alpha_weighted_loc_loss_over_gen_loss": (
                    (masked_loc_loss * alpha_loc_loss / masked_gen_loss).detach().cpu().numpy().item()
                    if masked_gen_loss.item() > 0 else None
                ),
                "loc_aux_loss": masked_loc_aux_loss.detach().cpu().numpy().item(),
                "alpha_loc_aux": alpha_loc_aux_loss.detach().cpu().numpy().item(),
                "alpha_weighted_loc_aux_loss": (masked_loc_aux_loss * alpha_loc_aux_loss).detach().cpu().numpy().item(),
                "alpha_weighted_loc_aux_loss_over_gen_loss": (
                    (masked_loc_aux_loss * alpha_loc_aux_loss / masked_gen_loss).detach().cpu().numpy().item()
                    if masked_gen_loss.item() > 0 else None
                ),
                "repa_loss": masked_repa_loss.detach().cpu().numpy().item(),
                "alpha_repa": alpha_repa_loss.detach().cpu().numpy().item(),
                "alpha_weighted_repa_loss": (masked_repa_loss * alpha_repa_loss).detach().cpu().numpy().item(),
                "alpha_weighted_repa_loss_over_gen_loss": (
                    (masked_repa_loss * alpha_repa_loss / masked_gen_loss).detach().cpu().numpy().item()
                    if masked_gen_loss.item() > 0 else None
                ),
                "loc_repa_loss": masked_loc_repa_loss.detach().cpu().numpy().item(),
                "alpha_loc_repa": alpha_loc_repa_loss.detach().cpu().numpy().item(),
                "alpha_weighted_loc_repa_loss": (masked_loc_repa_loss * alpha_loc_repa_loss).detach().cpu().numpy().item(),
                "alpha_weighted_loc_repa_loss_over_gen_loss": (
                    (masked_loc_repa_loss * alpha_loc_repa_loss / masked_gen_loss).detach().cpu().numpy().item()
                    if masked_gen_loss.item() > 0 else None
                ),
            }
        }


        if 0:
            route_results = inspect_loss_routes_with_autograd_grad(
                model=self,
                loss_dict={
                    "total_loss": total_loss,
                    "loc_loss": masked_loc_loss,
                    "gen_loss": masked_gen_loss,
                    "repa_loss": masked_repa_loss,
                    # "loc_aux_loss": masked_loc_aux_loss,
                    # "loc_repa_loss": masked_loc_repa_loss,
                },
            )
            print_loss_route_summary(route_results)
            print_optimizer_managed_param_summary(self)

            """--csgo_config csgo_configs/exp13.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp13_debug --num_train_epochs 100 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 2 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --lora_r 16"""

            """--csgo_config csgo_configs/exp11.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp11_debug --num_train_epochs 100 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 2 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --lora_r 16"""

        return CausalLMOutputs

    def forward_for_aux_loc_loss(
        self,
        sigmas,
        noisy_latents,
        noise_pred,
        aux_loc_input_ids,
        aux_loc_labels,
        attention_mask,
        und_image_map,
        gen_image,
        loc_map_image,
        grid_thw,
        i_s_pos,
        image_sizes,
        task_id,
        return_dict,
        actions,
        loss_mask,
        pred_pixels_input: Optional[torch.Tensor] = None,
    ):
        # vt_gc_enabled = getattr(self.model.vision_tower, "gradient_checkpointing", False)
        # lm_gc_enabled = getattr(self.model.language_model, "gradient_checkpointing", False)
        # action_gc_enabled = getattr(self.model.action_dit, "gradient_checkpointing", False)

        # if vt_gc_enabled:
        #     self.model.vision_tower.gradient_checkpointing = False
        # if lm_gc_enabled:
        #     self.model.language_model.gradient_checkpointing = False
        # if action_gc_enabled:
        #     self.model.action_dit.gradient_checkpointing = False

        if pred_pixels_input is None:
            pred_pixels_input, pred_pixels_norm = self._decode_pred_pixels_input_from_flow(
                sigmas=sigmas,
                noisy_latents=noisy_latents,
                noise_pred=noise_pred,
                und_image_map=und_image_map,
            )
        else:
            pred_pixels_norm = ((pred_pixels_input * 0.5) + 0.5).clamp(0.0, 1.0)

        # 4. Forward Localization Branch with Generated Image
        # [Important] Freeze Loc Branch weights to avoid updating them with noisy gradients
        # We only want gradients to flow back to the Image (pred_pixels_input)

        # 构造 Loc 分支输入
        # Und Image = Generated Fps Image (pred_pixels_input)
        # Aux Image = Original Map (und_image passed in forward, which is actually map for Gen task)
        # Prompt = Loc Prompt (aux_loc_input_ids)

        if getattr(self.config, "use_external_loc_model", False):
            if loc_map_image is None:
                raise RuntimeError("External loc aux loss requires loc_map_image in batch.")
            pred_pixels_external = self._normalize_for_external_loc_model(pred_pixels_norm.clamp(0.0, 1.0))
            actions_pred = self._forward_external_loc_model(pred_pixels_external, loc_map_image)
            if getattr(self.config, "external_loc_use_circular_loss", True):
                masked_loc_loss = self._compute_codex_loc_regression_loss(actions_pred, actions).to(torch.float32)
            else:
                masked_loc_loss = torch.nn.MSELoss()(actions_pred, actions).to(torch.float32)
        elif getattr(self.config, "use_codex_vit_regression_head", False):
            # 临时冻结 Action Dit
            self.model.vit_loc_fusion.requires_grad_(False)
            if getattr(self.config, "is_action_dit_projector", False):
                self.model.action_dit_norm.requires_grad_(False)
                self.model.action_dit_projector.requires_grad_(False)
            self.model.regression_loc_head.requires_grad_(False)

            self.model.vit_loc_fusion.eval()
            if getattr(self.config, "is_action_dit_projector", False):
                self.model.action_dit_norm.eval()
                self.model.action_dit_projector.eval()
            self.model.regression_loc_head.eval()

            # with torch.no_grad():
            if 1:
                actions_pred = self._forward_codex_vit_regression_head(
                    pred_pixels_input,
                    und_image_map,
                )
                if getattr(self.config, "loc_use_circular_loss", True):
                    masked_loc_loss = self._compute_codex_loc_regression_loss(actions_pred, actions).to(torch.float32)
                else:
                    masked_loc_loss = torch.nn.MSELoss()(actions_pred, actions).to(torch.float32)

            self.model.vit_loc_fusion.requires_grad_(True)
            if getattr(self.config, "is_action_dit_projector", False):
                self.model.action_dit_norm.requires_grad_(True)
                self.model.action_dit_projector.requires_grad_(True)
            self.model.regression_loc_head.requires_grad_(True)

            self.model.vit_loc_fusion.train()
            if getattr(self.config, "is_action_dit_projector", False):
                self.model.action_dit_norm.train()
                self.model.action_dit_projector.train()
            self.model.regression_loc_head.train()
        elif getattr(self.config, "use_vit_cls_regression_head", False):
            # 临时冻结 Action Dit
            if getattr(self.config, "is_action_dit_projector", False):
                self.model.action_dit_norm.requires_grad_(False)
                self.model.action_dit_projector.requires_grad_(False)
            self.model.regression_loc_head.requires_grad_(False)

            if getattr(self.config, "is_action_dit_projector", False):
                self.model.action_dit_norm.eval()
                self.model.action_dit_projector.eval()
            self.model.regression_loc_head.eval()

            # with torch.no_grad():
            if 1:
                actions_pred = self._forward_vit_regression_head(
                    pred_pixels_input,
                    und_image_map,
                )
                if getattr(self.config, "loc_use_circular_loss", True):
                    masked_loc_loss = self._compute_codex_loc_regression_loss(actions_pred, actions).to(torch.float32)
                else:
                    masked_loc_loss = torch.nn.MSELoss()(actions_pred, actions).to(torch.float32)

            if getattr(self.config, "is_action_dit_projector", False):
                self.model.action_dit_norm.requires_grad_(True)
                self.model.action_dit_projector.requires_grad_(True)
            self.model.regression_loc_head.requires_grad_(True)

            if getattr(self.config, "is_action_dit_projector", False):
                self.model.action_dit_norm.train()
                self.model.action_dit_projector.train()
            self.model.regression_loc_head.train()
        else:
            # 获取 Gen 任务原本的 Map 输入 (在 prepare_inputs 里它是 und_image)
            combined_und_images = torch.cat([pred_pixels_input, und_image_map], dim=0)
            # with torch.no_grad():
            if 1:
                ( # return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds, combined_img_idx, combined_image_embeds, bidr_attention_mask
                    aux_loc_input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    aux_loc_labels,
                    target_image_embeds, #latents
                    combined_img_idx,
                    combined_image_embeds,
                    bidr_attention_mask
                ) = self.prepare_inputs_labels_for_multimodal(
                    aux_loc_input_ids,
                    None, #position_ids,
                    attention_mask,
                    None, #past_key_values,
                    aux_loc_labels,
                    gen_image,
                    combined_und_images, # Pass und_image (which contains all input visuals)
                    None, #grid_thw,
                    None, #i_s_pos,
                    None, #image_sizes,
                    task_id,
                )
            und_img_idx = combined_img_idx[:combined_img_idx.size(0)//2, ...] #und_img_idx,sum()=32768 #32768/256=128.0
            aux_img_idx = combined_img_idx[combined_img_idx.size(0)//2:, ...]#aux_img_idx.sum()=tensor(14592, device='cuda:0') #14592/256=57
            und_image_embeds = combined_image_embeds[:combined_image_embeds.size(0)//2, ...]#torch.Size([128, 256, 896])
            aux_image_embeds = combined_image_embeds[combined_image_embeds.size(0)//2:, ...]#torch.Size([128, 256, 896])

            if getattr(self.config, "use_vit_regression_head", False):
                # 临时冻结 Action Dit
                if getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_norm.requires_grad_(False)
                    self.model.action_dit_projector.requires_grad_(False)
                self.model.regression_loc_head.requires_grad_(False)

                if getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_norm.eval()
                    self.model.action_dit_projector.eval()
                self.model.regression_loc_head.eval()

                # with torch.no_grad():
                if 1:
                    # und_feature = self.get_model().img_pooler(und_image_embeds)
                    # aux_feature = self.get_model().img_pooler(aux_image_embeds)
                    # und_feature = und_image_embeds[:, 0, :]
                    # aux_feature = aux_image_embeds[:, 0, :]
                    # concat_feature = torch.cat([und_feature, aux_feature], dim=-1)
                    concat_feature = self.get_model().cross_view_fusion(und_image_embeds, aux_image_embeds)
                    concat_feature = self.get_model().action_dit_norm(concat_feature)
                    if getattr(self.config, "is_action_dit_projector", False):
                        concat_feature = self.get_model().action_dit_projector(concat_feature)
                    actions_pred = self.get_model().regression_loc_head(concat_feature)
                    if getattr(self.config, "loc_use_circular_loss", True):
                        masked_loc_loss = self._compute_codex_loc_regression_loss(actions_pred, actions).to(torch.float32)
                    else:
                        masked_loc_loss = torch.nn.MSELoss()(actions_pred, actions).to(torch.float32)

                # 恢复训练 Action Dit
                if getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_norm.requires_grad_(True)
                    self.model.action_dit_projector.requires_grad_(True)
                self.model.regression_loc_head.requires_grad_(True)

                if getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_norm.train()
                    self.model.action_dit_projector.train()
                self.model.regression_loc_head.train()

            else:
                # --- B. Main LLM Forward (Understanding) ---
                position_ids = torch.cumsum(attention_mask, dim=1) - 1
                position_ids[position_ids < 0] = 0
                # bs=8显存占用=13314MiB
                # with torch.no_grad():
                if 1:
                    outputs = self.model.language_model(
                        attention_mask=attention_mask, #torch.Size([2, 617])
                        position_ids=position_ids, #torch.Size([2, 617])
                        inputs_embeds=inputs_embeds, #torch.Size([2, 617, 896])
                        output_hidden_states=True,
                        return_dict=return_dict, #True
                        use_cache=False
                    )

                # Last Hidden State from LLM: [BS, Seq_Len, Hidden_Size]
                # This contains contextualized features of both text and images.
                hidden_states = outputs.hidden_states[-1] #torch.Size([128, 707, 896])

                # Re-fill und_image embeddings (Skip Connection logic from UniLIP)
                if und_image_embeds is not None and und_img_idx is not None:
                    hidden_states[und_img_idx] = und_image_embeds.to(hidden_states.device).flatten(0,1)

                # 在传入forward_for_aux_loc_loss之前，已经将所有task_id转换为0，即该batch中所有样本都是loc任务
                is_loc_task = (task_id == 0)#is_loc_task.shape=torch.Size([128])#is_loc_task.sum()=tensor(57, device='cuda:0')
                if aux_image_embeds is not None and und_img_idx is not None: #aux_img_idx.sum()/128 = 57
                    hidden_states[aux_img_idx] = aux_image_embeds[is_loc_task].to(hidden_states.device).flatten(0,1)#hidden_states[aux_img_idx].shape=torch.Size([14592, 896]) #aux_image_embeds[is_loc_task].shape=torch.Size([2, 256, 896])

                # 临时冻结 Action Dit
                if getattr(self.config, "use_pi05_action_dit", False) and getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_connector.requires_grad_(False)
                if getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_norm.requires_grad_(False)
                    self.model.action_dit_projector.requires_grad_(False)
                self.model.action_dit.requires_grad_(False)
                self.model.action_in_proj.requires_grad_(False)
                self.model.action_out_proj.requires_grad_(False)
                self.model.time_mlp_in.requires_grad_(False)
                self.model.time_mlp_out.requires_grad_(False)
                if getattr(self.config, "use_pi05_action_dit", False) and getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_connector.eval()
                if getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_norm.eval()
                    self.model.action_dit_projector.eval()
                self.model.action_dit.eval()
                self.model.action_in_proj.eval()
                self.model.action_out_proj.eval()
                self.model.time_mlp_in.eval()
                self.model.time_mlp_out.eval()
                # bs=8显存占用=13314MiB
                # 5. Original LOCALIZATION Branch (Flow Matching Path)
                # with torch.no_grad():
                if 1:
                    actions = actions#torch.Size([128, 5])
                    noise = self.sample_noise(actions.shape, actions.device)#torch.Size([128, 5])
                    time = self.sample_time(actions.shape[0], actions.device)#torch.Size([128])
                    time_expanded = time[:, None, None].to(actions.dtype)
                    x_t = time_expanded * noise + (1 - time_expanded) * actions
                    u_t = noise - actions

                    suffix_emb, adarms_cond = self.embed_action_suffix(
                            x_t, #torch.Size([128, 1, 5])
                            time, #torch.Size([128])
                            llm_hidden_size=1024 if getattr(self.config, "use_pi05_action_dit", False) else self.model.config.text_config.hidden_size,
                            device=actions.device,
                            dtype=hidden_states.dtype
                        )
                    if getattr(self.config, "use_pi05_action_dit", False):
                        hidden_states = self.get_model().action_dit_connector(hidden_states)
                    hidden_states = self.get_model().action_dit_norm(hidden_states)


                    bs, seq_len, hidden_dim = hidden_states.shape
                    valid_lens = attention_mask.sum(dim=1).long()

                    action_dit_inputs = torch.cat([hidden_states, torch.zeros_like(suffix_emb)], dim=1)
                    target_indices = valid_lens.view(-1, 1, 1).expand(-1, 1, hidden_dim)
                    action_dit_inputs = action_dit_inputs.scatter(1, target_indices, suffix_emb)

                    action_dit_att_mask = torch.cat([attention_mask, torch.zeros((bs, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
                    mask_indices = valid_lens.view(-1, 1)
                    action_dit_att_mask = action_dit_att_mask.scatter(1, mask_indices, 1)

                    action_dit_pos_ids = torch.cat([position_ids, torch.zeros((bs, 1), device=position_ids.device, dtype=position_ids.dtype)], dim=1)
                    action_pos_ids = valid_lens.view(-1, 1)
                    # action_dit_pos_ids = action_dit_pos_ids.scatter(1, mask_indices, action_pos_ids)
                    current_seq_len = action_dit_pos_ids.shape[1]
                    range_ids = torch.arange(current_seq_len, device=position_ids.device).unsqueeze(0) # [1, Seq+1]
                    mask_after_valid = range_ids >= valid_lens.view(-1, 1)
                    action_dit_pos_ids = torch.where(
                        mask_after_valid,
                        action_pos_ids,
                        action_dit_pos_ids
                    )

                    if getattr(self.config, "is_action_dit_projector", False):
                        action_dit_inputs = self.get_model().action_dit_projector(action_dit_inputs)

                    if getattr(self.config, "use_pi05_action_dit", False):
                        suffix_output = self.get_model().action_dit(
                            inputs_embeds=action_dit_inputs,
                            attention_mask=self._prepare_attention_masks_4d_from_attn_masks_1d(action_dit_att_mask),
                            position_ids=action_dit_pos_ids,
                            use_cache=False,
                            adarms_cond=adarms_cond,
                        )
                        action_hidden = suffix_output.last_hidden_state
                    else:
                        if getattr(self.config, 'is_action_dit_dense_timestep', False):
                            action_outputs = self.action_dit_forward_with_adarmscond(
                                hidden_states=action_dit_inputs,
                                attention_mask=action_dit_att_mask,
                                position_ids=action_dit_pos_ids,
                                adarms_cond=adarms_cond,
                            )
                            action_hidden = action_outputs
                        else:
                            action_outputs = self.model.action_dit(
                                inputs_embeds=action_dit_inputs,
                                attention_mask=action_dit_att_mask,
                                position_ids=action_dit_pos_ids,
                                output_hidden_states=True,
                                return_dict=return_dict,
                                use_cache=False
                            )
                            # Get output corresponding to the Action Token (Last token)
                            # output: [BS, Seq+1, Hidden]
                            action_hidden = action_outputs.hidden_states[-1]# [BS, seq+1, Hidden] #[:, -1:, :] # [BS, 1, Hidden] #torch.Size([128, 1, 896])

                    # Gather from the same indices we scattered to
                    gather_indices = valid_lens.view(-1,1,1).expand(-1, -1, hidden_dim)
                    action_hidden = action_hidden.gather(1, gather_indices) # [BS, 1, Hidden]

                    v_t_pred = self.get_model().action_out_proj(action_hidden) # [BS, 1, 5] #torch.Size([128, 1, 5])
                    loc_loss = F.mse_loss(v_t_pred.float(), u_t.float(), reduction="none") #torch.Size([128, 1, 5])
                    loc_loss = loc_loss.mean(dim=[1, 2]) # [BS] #torch.Size([128])

                    # 加权：只在 t 小的时候 (生成接近完成) 计算 Loss
                    # sigmas 越大噪声越大。我们希望 sigma 小的时候权重高。
                    weight = (1.0 - sigmas).clamp(min=0)
                    loc_loss = (loc_loss * weight.squeeze())

                    # Apply Mask: Only count aux loc loss for Gen samples
                    masked_loc_loss = (loc_loss * loss_mask[:, 1]).mean()

                # 恢复训练 Action Dit
                if getattr(self.config, "use_pi05_action_dit", False) and getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_connector.requires_grad_(True)
                if getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_norm.requires_grad_(True)
                    self.model.action_dit_projector.requires_grad_(True)
                self.model.action_dit.requires_grad_(True)
                self.model.action_in_proj.requires_grad_(True)
                self.model.action_out_proj.requires_grad_(True)
                self.model.time_mlp_in.requires_grad_(True)
                self.model.time_mlp_out.requires_grad_(True)
                if getattr(self.config, "use_pi05_action_dit", False) and getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_connector.train()
                if getattr(self.config, "is_action_dit_projector", False):
                    self.model.action_dit_norm.train()
                    self.model.action_dit_projector.train()
                self.model.action_dit.train()
                self.model.action_in_proj.train()
                self.model.action_out_proj.train()
                self.model.time_mlp_in.train()
                self.model.time_mlp_out.train()

        # # =================================================================
        # # [修复结束] 无论如何，恢复 Gradient Checkpointing 的原始状态
        # # =================================================================
        # if vt_gc_enabled:
        #     self.model.vision_tower.gradient_checkpointing = True
        # if lm_gc_enabled:
        #     self.model.language_model.gradient_checkpointing = True
        # if action_gc_enabled:
        #     self.model.action_dit.gradient_checkpointing = True

        # bs=8显存占用=13316MiB
        return masked_loc_loss

    def action_dit_forward_with_adarmscond(
        self,
        hidden_states,#torch.Size([128, 708, 896])
        attention_mask, #torch.Size([128, 708])
        position_ids, #torch.Size([128, 708])
        adarms_cond,
        # 和Pi05的不同：除了没有使用adarms_cond之外，action_dit也没有使用full_att_2d_masks_4d
            # Pi05的language_model和DiT都使用了prefix双向，suffix单向的mask；
            # 而UniLIP的language_model和DiT使用了单向mask，但是中间的llm_connector使用了单向mask；

    ):
        # 1. 准备 Rotary Embedding (Qwen2 需要)
        kv_seq_len = hidden_states.shape[1]
        # cos, sin = self.model.action_dit.rotary_emb(hidden_states, position_ids)
        position_embeddings = self.model.action_dit.rotary_emb(hidden_states, position_ids)

        # [可选] 在 Loop 之前
        # 利用 transformers 提供的 helper 扩展 mask
        # 注意：Qwen2 的实现可能需要 4D mask
        # extended_attention_mask = self.model.action_dit.get_extended_attention_mask(
        #     attention_mask,
        #     attention_mask.shape,
        #     attention_mask.device
        # )# 在 layer.self_attn 中使用 extended_attention_mask

        # from transformers.masking_utils import create_causal_mask
        # mask_kwargs = {
        #         "config": self.model.action_dit.config,
        #         "input_embeds": hidden_states,
        #         "attention_mask": attention_mask,
        #         "cache_position": None,
        #         "past_key_values": None,
        #         "position_ids": position_ids,
        #     }
        # extended_attention_mask = create_causal_mask(**mask_kwargs)
        # ============================================================
        # [修复] 使用 Qwen2 内部的 _update_causal_mask
        # ============================================================
        # 这个方法会自动处理：
        # 1. Causal Mask (下三角)
        # 2. Padding Mask (根据传入的 attention_mask)
        # 3. 维度扩展 -> [BS, 1, Seq, Seq]
        # 4. Flash Attention 兼容性 (如果是 FA2，它可能返回 None 或特定格式)

        # 为了兼容 transformers 版本，我们需要构造 cache_position (通常是 arange)
        # 如果报错缺少 cache_position，可以用下面的简单逻辑生成
        # cache_position = torch.arange(
        #     0, kv_seq_len, device=hidden_states.device
        # )
        # causal_mask = self.model.action_dit._update_causal_mask(
        #     attention_mask,
        #     hidden_states,
        #     cache_position,
        #     past_key_values=None,
        #     output_attentions=False
        # )
        # 备用方案：手动构建 4D Causal Mask
        batch_size, seq_len = hidden_states.shape[:2]

        # 1. 创建下三角 Causal Mask [Seq, Seq]
        # min_dtype 是 float 的最小值 (e.g. -65504 for fp16, -3.4e38 for fp32)
        min_dtype = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.full((seq_len, seq_len), min_dtype, device=hidden_states.device, dtype=hidden_states.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1) # 上三角为负无穷，下三角为0

        # 2. 扩展维度 [1, 1, Seq, Seq]
        causal_mask = causal_mask[None, None, :, :]

        # 3. 处理 Padding Mask [BS, Seq] -> [BS, 1, 1, Seq]
        # 注意：attention_mask 是 1=Valid, 0=Pad
        # 我们需要把 0 变成负无穷，1 变成 0
        padding_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype)
        padding_mask = padding_mask.masked_fill(attention_mask == 0, min_dtype)
        padding_mask = padding_mask[:, None, None, :]

        # 4. 合并 [BS, 1, Seq, Seq]
        # 利用广播机制：Causal (mask future) + Padding (mask pad tokens)
        combined_mask = causal_mask + padding_mask

        # 2. 逐层前向传播
        for i, layer in enumerate(self.model.action_dit.layers):

            # --- Attention Block ---
            residual = hidden_states

            # [关键] 调用 AdaRMS Input Norm (传入 cond)
            # adarms_cond 是我们在 embed_action_suffix 里算出来的
            hidden_states, gate_attn = layer.input_layernorm(hidden_states, cond=adarms_cond)

            # Self Attention
            # Qwen2 的 self_attn forward 签名通常是:
            # (hidden_states, attention_mask, position_ids, past_key_values, output_attentions, use_cache, **kwargs)
            # 注意：我们需要手动处理 cos/sin 的传入，Qwen2 内部通常会自动处理，但我们需要确保 position_ids 正确
            hidden_states, _ = layer.self_attn(
                hidden_states=hidden_states,#torch.Size([2, 708, 896])
                attention_mask=combined_mask,#causal_mask,#extended_attention_mask[:, None, :, :], # 需要广播为 [BS, 1, Seq, Seq] 或者是 FlashAttn 格式
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = residual + gate_attn * hidden_states
            # --- MLP Block ---
            residual = hidden_states
            # [关键] 调用 AdaRMS Post Norm (传入 cond)
            hidden_states, gate_mlp = layer.post_attention_layernorm(hidden_states, cond=adarms_cond)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + gate_mlp * hidden_states

        # 3. Final Norm
        hidden_states, _ = self.model.action_dit.norm(hidden_states, cond=adarms_cond)
        # 4. 提取输出 (保持原逻辑)
        action_outputs = hidden_states

        return action_outputs

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        tokenizer: AutoTokenizer,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_var: Optional[float] = None,
        generator=None,
        guidance_scale: float = 4.5,
    ):
        dit_path = self.model.config.dit_path
        scheduler = DPMSolverMultistepScheduler.from_pretrained(dit_path, subfolder="scheduler")

        N_QUERY = self.get_n_query()
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        device = self.get_model().device
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)  # B x N

        text_embeds = self.get_model().language_model.embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)

        if pixel_values is not None:
            und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
            pixel_values = pixel_values.type(self.vision_tower.dtype)
            vision_feature_layer = self.config.vision_feature_layer
            vision_feature_select_strategy = self.config.vision_feature_select_strategy
            und_image_embeds = self.model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=None,
            )

            text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device).repeat(2,1,1).flatten(0,1)

        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1).int()
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        outputs = self.model.language_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask.bool(),
            position_ids= position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]
        if pixel_values is not None:
            und_image_idx = torch.cat([und_image_idx, torch.zeros_like(latent_queries[:, :, 0]).bool()], dim=1)
            hidden_states[und_image_idx] = und_image_embeds.to(hidden_states.device).repeat(2,1,1).flatten(0,1)

        attention_mask = attention_mask.bool()
        bidr_attention_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
        bidr_attention_mask = bidr_attention_mask.unsqueeze(1)
        bidr_attention_mask = (1-bidr_attention_mask.float())*-100000

        img_hidden_states = self.model.llm_connector(
            inputs_embeds=hidden_states,
            attention_mask=bidr_attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]

        img_hidden_states = self.get_model().projector(img_hidden_states)

        steps = 20
        logging.info("steps, guiance scale", steps, guidance_scale)
        # null first
        bsz = img_hidden_states.shape[0]
        img_hidden_states = torch.cat([img_hidden_states[bsz//2:], img_hidden_states[:bsz//2]])
        attention_mask = torch.cat([attention_mask[bsz//2:], attention_mask[:bsz//2]])

        output_img = self.sample_image_latents(img_hidden_states, scheduler, encoder_attention_mask=attention_mask, generator=generator, num_inference_steps=steps, guidance_scale=guidance_scale)

        logging.info(self.model.config.unilip_factor)
        output_img = self.model.vae_decoder.vae_decode(output_img.float()/ self.model.config.unilip_factor)

        output_img = ((output_img[0].permute(1,2,0).clamp(-1,1).float().cpu().numpy() + 1)/2)

        original_height, original_width = output_img.shape[:2]
        # 定义缩放比例
        scale_factor = 28 / 32  # 0.875，即缩小为原来的87.5%
        # 计算新的尺寸
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        new_size = (new_width, new_height)
        # 使用双线性插值（INTER_LINEAR）进行缩放
        resized_img = cv2.resize(output_img, new_size, interpolation=cv2.INTER_LINEAR)

        return resized_img

    def sample_image_latents(
        self,
        img_hidden_states,
        scheduler,
        encoder_attention_mask=None,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        **kwargs,
    ):
        device = img_hidden_states.device
        dtype = img_hidden_states.dtype

        img_hidden_states_input = img_hidden_states
        # here already is cfg feature
        batch_size = img_hidden_states.shape[0]//2
        latent_size = 16
        latent_channels = 32

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # set step values
        if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        else:
            scheduler.set_timesteps(num_inference_steps)

        # Repeat z_latents and conditions for each image per prompt
        img_hidden_states_input = img_hidden_states_input.repeat_interleave(num_images_per_prompt, dim=0)

        for t in scheduler.timesteps:
            latent_model_input = latents.repeat(2, 1, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(img_hidden_states_input.dtype)

            # predict noise model_output
            noise_pred = self.get_model().dit(
                latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                encoder_hidden_states=img_hidden_states_input,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

        return latents

    # # ==========================================
    # # 4. [NEW] Inference: Generate Action (Flow Matching)
    # # ==========================================
    # @torch.no_grad()
    # def generate_action(
    #     self,
    #     text: List[str],
    #     tokenizer: AutoTokenizer,
    #     und_images: Optional[torch.Tensor] = None,
    #     aux_images: Optional[torch.Tensor] = None,
    #     num_steps: int = 10
    # ):
    #     """
    #     Run inference for Localization task using Flow Matching Euler Solver.
    #     Analogous to Pi0.5 `sample_actions`.
    #     """
    #     # 1. Precompute Context (LLM Forward)
    #     inputs = tokenizer(text, padding="longest", return_tensors="pt")
    #     device = self.get_model().device
    #     attention_mask = inputs.attention_mask.to(device)
    #     input_ids = inputs.input_ids.to(device)  # B x N
    #     text_embeds = self.get_model().language_model.embed_tokens(input_ids)

    #     vision_feature_layer = self.config.vision_feature_layer
    #     vision_feature_select_strategy = self.config.vision_feature_select_strategy

    #     if und_images is not None:
    #         und_image_embeds = self.model.get_image_features(
    #             pixel_values=und_images,
    #             vision_feature_layer=vision_feature_layer,
    #             vision_feature_select_strategy=vision_feature_select_strategy,
    #             image_sizes=None,
    #         )

    #     if aux_images is not None:
    #         aux_image_embeds = self.model.get_image_features(
    #             pixel_values=aux_images,
    #             vision_feature_layer=vision_feature_layer,
    #             vision_feature_select_strategy=vision_feature_select_strategy,
    #             image_sizes=None,
    #         )

    #     und_image_idx, aux_image_idx = split_image_tokens(input_ids, IMAGE_TOKEN_IDX)
    #     if und_images is not None and und_image_idx.any():
    #         text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device).repeat(2,1,1).flatten(0,1)
    #     if aux_images is not None and aux_image_idx.any():
    #         text_embeds[aux_image_idx] = aux_image_embeds.to(text_embeds.device).repeat(2,1,1).flatten(0,1)

    #     attention_mask = torch.cat([attention_mask, torch.ones_like(x_t[:, :, 0])], dim=1).int()
    #     position_ids = torch.cumsum(attention_mask, dim=1) - 1
    #     position_ids[position_ids < 0] = 0

    #     # Forward Context (LLM)
    #     outputs = self.model.language_model(
    #         inputs_embeds=text_embeds,
    #         attention_mask=attention_mask.bool(),
    #         position_ids=position_ids,
    #         output_hidden_states=True,
    #         return_dict=True,
    #         use_cache=True # Enable KV Cache for efficiency?
    #         # Note: Action Connector might not share KV cache structure easily if distinct models.
    #         # Pi0.5 uses cache for Prefix. Here Context is separate model.
    #         # We can cache the Context Output (Hidden States).
    #     )
    #     hidden_states = outputs.hidden_states[-1] # [BS, Seq, Hidden]

    #     if und_images is not None and und_image_idx.any():
    #         hidden_states[und_image_idx] = und_image_embeds.to(hidden_states.device).repeat(2,1,1).flatten(0,1)
    #     if aux_images is not None and aux_image_idx.any():
    #         hidden_states[aux_image_idx] = aux_image_embeds.to(hidden_states.device).repeat(2,1,1).flatten(0,1)

    #     # Pre-prepare Context inputs for Action DiT
    #     # Since Action DiT is just layers, we concat inputs every step.
    #     # Optimization: Pi0 uses KV cache for prefix.
    #     # Here, `action_dit` is a separate module instance.
    #     # We can treat `hidden_states` as the "Prefix Embeddings".

    #     # 2. Initialize Noise
    #     bsize = input_ids.shape[0]
    #     action_dim = self.model.config.action_dim
    #     noise = self.sample_noise((bsize, 1, action_dim), device=device)

    #     # 3. Euler Solver Loop

    #     dt = -1.0 / num_steps
    #     dt = torch.tensor(1.0, device=device, dtype=torch.float32)

    #     x_t = noise
    #     t = torch.tensor(1.0, dtype=torch.float32, device=device)
    #     while t >= -dt / 2: # Stop near 0
    #         # Embed Suffix
    #         expanded_time = t.expand(bsize)
    #         suffix_emb, adarms_cond = self.embed_action_suffix(
    #             x_t,
    #             expanded_time,
    #             llm_hidden_size=self.model.config.text_config.hidden_size,
    #             device=device,
    #             dtype=hidden_states.dtype
    #         )

    #         # Concat
    #         action_dit_inputs = torch.cat([hidden_states, suffix_emb], dim=1)

    #         # Masks & Pos IDs (Same as Forward)
    #         action_mask = torch.ones((bsize, 1), device=device, dtype=attention_mask.dtype)
    #         action_dit_att_mask = torch.cat([attention_mask, action_mask], dim=1)
    #         action_pos_id = position_ids.max(dim=1)[0].unsqueeze(1) + 1
    #         action_dit_pos_ids = torch.cat([position_ids, action_pos_id], dim=1)

    #         # Forward Connector
    #         action_out = self.model.action_dit(
    #             inputs_embeds=action_dit_inputs,
    #             attention_mask=action_dit_att_mask,
    #             position_ids=action_dit_pos_ids,
    #             output_hidden_states=True,
    #             # adarms_cond=[None, adarms_cond],
    #             # 和Pi05的不同：除了没有使用adarms_cond之外，action_dit也没有使用full_att_2d_masks_4d
    #                 # Pi05的language_model和DiT都使用了prefix双向，suffix单向的mask；
    #                 # 而UniLIP的language_model和DiT使用了单向mask，但是中间的llm_connector使用了单向mask；
    #         )

    #         # Predict Velocity
    #         action_feat = action_out.hidden_states[-1][:, -1:, :]
    #         v_t = self.get_model().action_out_proj(action_feat)

    #         # Euler Step
    #         x_t = x_t + dt * v_t
    #         t += dt

    #     return x_t # Final Denoised Action [BS, 1, 5]



    # ==========================================
    # 4. [NEW] Inference: Generate Action (Flow Matching)
    # ==========================================
    @torch.no_grad()
    def generate_action2(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        und_image: Optional[torch.Tensor] = None,
        aux_image: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        generator: Optional[torch.Generator] = None
    ):
        """
        Run inference for Localization task using Flow Matching Euler Solver.
        Adapts the logic from `forward` (Right-Shift & Fill) to ensure consistency.
        """
        if getattr(self.config, "use_codex_vit_regression_head", False):
            return  self._forward_codex_vit_regression_head(und_image, aux_image,)
        elif getattr(self.config, "use_vit_cls_regression_head", False):
            return self._forward_vit_regression_head(und_image, aux_image)

        # 1. Get Vision Features
        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        vision_dtype = self.model.vision_tower.embeddings.patch_embedding.weight.dtype

        und_image_embeds = None
        if und_image is not None:
            und_image_embeds = self.model.get_image_features(
                pixel_values=und_image.to(dtype=vision_dtype),
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        aux_image_embeds = None
        if aux_image is not None:
            aux_image_embeds = self.model.get_image_features(
                pixel_values=aux_image.to(dtype=vision_dtype),
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        if getattr(self.config, "use_vit_regression_head", False):
            # und_feature = self.get_model().img_pooler(und_image_embeds)
            # aux_feature = self.get_model().img_pooler(aux_image_embeds)
            # und_feature = und_image_embeds[:, 0, :]
            # aux_feature = aux_image_embeds[:, 0, :]
            # concat_feature = torch.cat([und_feature, aux_feature], dim=-1)
            concat_feature = self.get_model().cross_view_fusion(und_image_embeds, aux_image_embeds)
            concat_feature = self.get_model().action_dit_norm(concat_feature)
            if getattr(self.config, "is_action_dit_projector", False):
                concat_feature = self.get_model().action_dit_projector(concat_feature)
            actions_pred = self.get_model().regression_loc_head(concat_feature)

            return actions_pred

        else:
            # 2. Embed Text & Replace Image Tokens
            text_embeds = self.get_model().language_model.embed_tokens(input_ids)
            und_image_idx, aux_image_idx = split_image_tokens(input_ids, IMAGE_TOKEN_IDX)

            # 3. Replace Image Features
            if und_image is not None and und_image_idx.any():
                # Broadcast embeddings to batch size if needed (e.g. 1 image for all prompts)
                # Assuming standard [BS, C, H, W] input for simplicity based on collator
                # If batch sizes match, no repeat needed. If single image for batch, repeat.
                if und_image_embeds.shape[0] == 1 and text_embeds.shape[0] > 1:
                    und_image_embeds = und_image_embeds.repeat(text_embeds.shape[0], 1, 1)
                text_embeds[und_image_idx] = und_image_embeds.flatten(0,1)

            if aux_image is not None and aux_image_idx.any():
                if aux_image_embeds.shape[0] == 1 and text_embeds.shape[0] > 1:
                    aux_image_embeds = aux_image_embeds.repeat(text_embeds.shape[0], 1, 1)
                text_embeds[aux_image_idx] = aux_image_embeds.flatten(0,1)

            # 4. Prepare Context Position IDs
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            position_ids[position_ids < 0] = 0

            # 5. Forward Context (LLM)
            # We need the last hidden state as the "Context" for the Action Head
            outputs = self.model.language_model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask.bool(),
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False # Inference usually benefits from cache, but here we just need the last state once
            )
            # [BS, Seq, Hidden]
            hidden_states = outputs.hidden_states[-1]

            # Re-fill image embeddings (Skip Connection logic)
            # Important: indices must match flattened structure or batch structure
            if und_image_embeds is not None and und_image_idx.any():
                hidden_states[und_image_idx] = und_image_embeds.flatten(0,1)
            if aux_image_embeds is not None and aux_image_idx.any():
                hidden_states[aux_image_idx] = aux_image_embeds.flatten(0,1)

            if getattr(self.config, "use_pi05_action_dit", False):
                hidden_states = self.get_model().action_dit_connector(hidden_states)

            # 6. action_dit_norm
            if not getattr(self.config, 'is_exp5_eval_without_aciton_dit_premodules', False):
                hidden_states = self.get_model().action_dit_norm(hidden_states)

            # 7. Initialize Flow Matching Loop
            bsize = input_ids.shape[0]
            if getattr(self.config, "use_pi05_action_dit", False):
                action_dim = self.get_model()._default_pi05_action_dim
            else:
                action_dim = self.model.config.action_dim

            # Sample initial noise x_1
            noise = self.sample_noise((bsize, 1, action_dim), device=hidden_states.device)

            # dt tensor for calculation
            dt = -1.0 / num_steps
            dt = torch.tensor(dt, device=hidden_states.device, dtype=hidden_states.dtype)

            # Calculate Valid Lengths for Right-Shift Insertion
            valid_lens = attention_mask.sum(dim=1, keepdim=True).long() # [BS, 1]
            bs, seq_len, hidden_dim = hidden_states.shape

            # 8. Euler Solver Loop
            # Stop at t=0 (or close to it, Pi0 uses -dt/2 for safety)
            x_t = noise
            t = torch.tensor(1.0, device=hidden_states.device, dtype=hidden_states.dtype)
            while t >= -dt / 2:
                # 8.1. Embed Suffix (Noisy Action + Time)
                expanded_time = t.expand(bsize)
                suffix_emb, adarms_cond = self.embed_action_suffix(
                    x_t,
                    expanded_time,
                    llm_hidden_size=1024 if getattr(self.config, "use_pi05_action_dit", False) else self.model.config.text_config.hidden_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )

                # 8.2. Construct Action Inputs (Right Shift & Fill Strategy)
                # Reusing the robust logic from forward pass to avoid NaN

                # 8.2.1. Inputs: Concat Hidden + Last Token (Safe Padding)
                # last_token_states = hidden_states[:, -1:, :]
                # extended_inputs = torch.cat([hidden_states, last_token_states], dim=1)
                # 8.2.1. Inputs: Concat Hidden + Zero Token
                extended_inputs = torch.cat([hidden_states, torch.zeros_like(suffix_emb)], dim=1)
                # Scatter Suffix to valid positions
                scatter_indices = valid_lens.unsqueeze(-1).expand(-1, -1, hidden_dim)
                action_dit_inputs = extended_inputs.scatter(1, scatter_indices, suffix_emb)

                # 8.2.2. Mask: Extend and Set True at Action Position
                extended_mask = torch.cat([
                    attention_mask,
                    torch.zeros((bs, 1), device=hidden_states.device, dtype=attention_mask.dtype)
                ], dim=1)
                mask_indices = valid_lens.view(-1, 1)
                action_dit_att_mask = extended_mask.scatter(1, mask_indices, 1)

                # 8.2.3. Position IDs: Extend and Set to Valid Length Index
                extended_pos_ids = torch.cat([
                    position_ids,
                    torch.zeros((bs, 1), device=hidden_states.device, dtype=position_ids.dtype)
                ], dim=1)
                action_pos_ids = valid_lens.view(-1, 1)
                if getattr(self.config, 'is_exp5_eval_without_aciton_dit_premodules', False):
                    action_dit_pos_ids = extended_pos_ids.scatter(1, mask_indices, action_pos_ids)
                else:
                    current_seq_len = extended_pos_ids.shape[1]
                    range_ids = torch.arange(current_seq_len, device=position_ids.device).unsqueeze(0) # [1, Seq+1]
                    # 生成掩码：如果当前位置 index >= valid_lens，则为 True
                    # [BS, 1] vs [1, Seq+1] -> Broadcast -> [BS, Seq+1]
                    mask_after_valid = range_ids >= valid_lens.view(-1, 1)
                    # 使用 torch.where 进行批量填充,保证pos_ids在valid_lens以后的位置继承action_pos_ids的值作为padding_pos_ids
                    # 逻辑：Mask 为 True 的地方填入 action_pos_id，False 的地方保持原样
                    extended_pos_ids = torch.where(
                        mask_after_valid,
                        action_pos_ids,      # 广播填充 [BS, 1] -> [BS, Mask区域]
                        extended_pos_ids  # 保持原值
                    )

                # 8.3 action_dit_projector
                if not getattr(self.config, 'is_exp5_eval_without_aciton_dit_premodules', False):
                    if getattr(self.config, 'is_action_dit_projector', False):
                        action_dit_inputs = self.get_model().action_dit_projector(action_dit_inputs)

                # 8.4. Forward Action DiT
                if getattr(self.config, "use_pi05_action_dit", False):
                        suffix_output = self.get_model().action_dit.forward(
                            inputs_embeds=action_dit_inputs,
                            attention_mask=self._prepare_attention_masks_4d_from_attn_masks_1d(action_dit_att_mask),
                            position_ids=extended_pos_ids,
                            use_cache=False,
                            adarms_cond=adarms_cond,
                        )
                        all_hidden_states = suffix_output.last_hidden_state
                else:
                    if getattr(self.config, 'is_action_dit_dense_timestep', False):
                        # Use custom forward loop with AdaRMS
                        action_outputs = self.action_dit_forward_with_adarmscond(
                            hidden_states=action_dit_inputs,
                            attention_mask=action_dit_att_mask,
                            position_ids=extended_pos_ids,
                            adarms_cond=adarms_cond
                        )
                        # Note: custom forward returns hidden_states directly
                        all_hidden_states = action_outputs
                    else:
                        # Use standard model forward
                        action_outputs = self.model.action_dit(
                            inputs_embeds=action_dit_inputs,
                            attention_mask=action_dit_att_mask,
                            position_ids=extended_pos_ids,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=False
                        )
                        all_hidden_states = action_outputs.hidden_states[-1]

                # 8.5. Extract Action Token Output
                # Gather from the same indices we scattered to
                gather_indices = valid_lens.unsqueeze(-1).expand(-1, -1, hidden_dim)
                action_feat = all_hidden_states.gather(1, gather_indices) # [BS, 1, Hidden]

                # 8.6. Predict Velocity & Step
                v_t = self.get_model().action_out_proj(action_feat)
                x_t = x_t + dt * v_t

                t += dt

            if getattr(self.config, "use_pi05_action_dit", False):
                return x_t[ : , : , :self.model.config.action_dim] # Final Denoised Action [BS, 1, 5]
            else:
                return x_t


AutoConfig.register("unified_unilip", Unified_UniLIP_InternVLConfig)
AutoModelForCausalLM.register(Unified_UniLIP_InternVLConfig, Unified_UniLIP_InternVLForCausalLM)
