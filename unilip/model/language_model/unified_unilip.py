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

            # # 输入action_dit前的feature首先进行normalize
            # self.action_dit_norm = Qwen2RMSNorm(llm_hidden_size, eps=1e-6)

            # 1. Action Dit
            # 这里的 config.action_dit_layer 可以在 model_args 中定义，默认比如 3 或 6
            action_layers = getattr(config, "action_dit_layer", 3)
            # 同样使用 InternVL 的后几层切片 (复用 llm_connector 的思路)
            internvl_model2 = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            self.action_dit = copy.deepcopy(internvl_model2.language_model)#.to(torch.bfloat16)
            del self.action_dit.layers[:-action_layers]
            del self.action_dit.embed_tokens # 不需要 Embedding 层，直接吃 Hidden States

            # 2. Flow Matching Heads
            # 将 Action (5D Pose) 映射到 LLM Hidden Size
            self.action_dim = getattr(config, "action_dim", 5) # x, y, z, pitch, yaw
            # self.action_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)#.to(torch.bfloat16)
            self.action_in_proj = nn.Linear(self.action_dim, llm_hidden_size)#.to(torch.bfloat16)

            # 3. 时间步 MLP
            self.time_mlp_in = nn.Linear(llm_hidden_size, llm_hidden_size)#.to(torch.bfloat16)
            self.time_mlp_out = nn.Linear(llm_hidden_size, llm_hidden_size)#.to(torch.bfloat16)

            # 4. 输出投影 (Hidden -> Action Velocity)
            self.action_out_proj = nn.Linear(llm_hidden_size, self.action_dim)#.to(torch.bfloat16)

            self.action_in_proj.apply(init_weights)
            self.time_mlp_in.apply(init_weights)
            self.time_mlp_out.apply(init_weights)
            self.action_out_proj.apply(init_weights)


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


    def initialize_localization_modules(self, model_args):
        # [Simulating previous code structure for brevity]
        self.config.action_horizon = getattr(model_args, 'action_horizon', 1) # CS2 Pose is typically single step
        self.config.action_dim = getattr(model_args, 'action_dim', 5)
        self.config.is_action_dit_dense_timestep = getattr(model_args, 'is_action_dit_dense_timestep', False)
        llm_hidden_size = self.multi_modal_projector[-1].weight.shape[-1]
        # [NEW] Initialize Action Connector & Heads
        # if getattr(self, 'action_dit', None) is None:
        if 1:
            # self.action_dit_norm = Qwen2RMSNorm(llm_hidden_size, eps=1e-6)

            path = model_args.mllm_hf_path
            logging.info(f"Initializing Action Connector from {path} slice...")
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

        # Enable Gradients for Action Path
        for p in self.action_dit.parameters(): p.requires_grad = True
        for p in self.action_in_proj.parameters(): p.requires_grad = True
        for p in self.time_mlp_in.parameters(): p.requires_grad = True
        for p in self.time_mlp_out.parameters(): p.requires_grad = True
        for p in self.action_out_proj.parameters(): p.requires_grad = True

        self.action_in_proj.apply(init_weights)
        self.time_mlp_in.apply(init_weights)
        self.time_mlp_out.apply(init_weights)
        self.action_out_proj.apply(init_weights)
        logging.info("Action VAE weights initialized successfully!")


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
        return self.get_model().get_vision_tower()

    def get_n_query(self):
        return self.get_model().config.n_query

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
                )
                # (B, HW, C) -> (B, C, H, W), assume H==W
                prompt_image_embeds = self.model.vae_decoder.clip_down(prompt_image_embeds)
            target_image_embeds = torch.clone(prompt_image_embeds).detach()
            target_image_embeds = target_image_embeds.mul_(self.model.unilip_factor) #torch.Size([128, 32, 16, 16])

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
        und_image_idx, aux_image_idx = split_image_tokens(input_ids, IMAGE_TOKEN_IDX)#und_image_idx=torch.Size([128, 707]) #aux_image_idx=torch.Size([128, 707])
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
        aux_img_idx = torch.logical_and(input_indicator, aux_image_idx)
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

    def get_model(self):
        return self.model

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
        actions: Optional[torch.FloatTensor] = None,   # [NEW] GT 5D Pose [BS, 1, 5]  # torch.Size([128, 5])
        loss_mask: Optional[torch.FloatTensor] = None, # [NEW] [BS, 2] -> [Loc_Weight, Gen_Weight]  # torch.Size([128, 2])
        aux_loc_input_ids: torch.LongTensor = None,
        aux_loc_labels: Optional[torch.LongTensor] = None,

        # Others
        grid_thw: Optional[torch.FloatTensor] = None, # None
        image_sizes: Optional[List[List[int]]] = None, # None
        return_dict: Optional[bool] = None, # None
        task_id: Optional[torch.FloatTensor] = None, # [NEW] [BS] -> 0: Loc, 1: Gen #torch.Size([128])
        **kwargs #dict_keys(['map_id', 'raw_prompt', 'map_name', 'pose_dict', 'num_items_in_batch']) #'num_items_in_batch': tensor(18631, device='cuda:0')
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- A. Input Preparation ---
        # Note: We merge und_image and aux_image logic.
        # In Dataset: Localization task has `und_image`(FPS) and `aux_image`(Map).
        # In Dataset: Generation task has `und_image`(Map) and `aux_image`(Empty).
        # We need to concat them for processing if both exist.

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
            ( # return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds, combined_img_idx, combined_image_embeds, bidr_attention_mask
                input_ids,
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
                gen_image,
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

        # --- B. Main LLM Forward (Understanding) ---
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

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
            # Branch 1: GENERATION (DiT Path)
            # ==========================================
            if loss_mask[:, 1].sum() > 0: # If any sample needs Generation
                is_gen_task = (task_id == 1)#is_gen_task.sum()=tensor(71, device='cuda:0')
                # TODO 如果这里使用[is_gen_task]过滤后再进行llm_connector和dit，能节约显存，但是由于[is_gen_task]的数目不一定恰好等于2的次方，所以可能会影响cuda加速运算。除非数据集collactor手动设置loc:gen=64:64。
                # 1. Process features via LLM Connector
                img_hidden_states = self.model.llm_connector(
                    attention_mask=bidr_attention_mask,#torch.Size([128, 1, 707, 707])
                    position_ids=position_ids,#torch.Size([128, 707])
                    inputs_embeds=hidden_states,#torch.Size([128, 707, 896])
                    output_hidden_states=True,
                    return_dict=return_dict,#True
                    use_cache=False
                ).hidden_states[-1]

                # 2. Project to DiT Caption Channel
                img_hidden_states = self.get_model().projector(img_hidden_states) #torch.Size([128, 707, 2304])

                # 3. Calculate DiT Loss
                if target_image_embeds is not None:#target_image_embeds=torch.Size([128, 32, 16, 16])
                    latents = target_image_embeds # [BS_Gen, C, H, W]
                    bsz = latents.shape[0] #128
                    noise = torch.randn_like(latents, device=latents.device)#torch.Size([128, 32, 16, 16])
                    u = compute_density_for_timestep_sampling(weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0) #torch.Size([128])
                    indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long() #torch.Size([128])
                    timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)#torch.Size([128])
                    sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)#torch.Size([128, 1, 1, 1])

                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise #torch.Size([128, 32, 16, 16])

                    # Forward DiT
                    noise_pred = self.get_model().dit(
                        noisy_latents, #torch.Size([128, 32, 16, 16])
                        timestep=timesteps, #128
                        encoder_hidden_states=img_hidden_states, # [BS, Seq, C] ##torch.Size([128, 707, 2304])
                        encoder_attention_mask=attention_mask, #torch.Size([128, 707])
                        return_dict=False
                    )[0] #torch.Size([128, 32, 16, 16])

                    target = noise - latents #torch.Size([128, 32, 16, 16])
                    # Compute raw MSE loss per sample [BS, ...]
                    gen_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")#torch.Size([128, 32, 16, 16])
                    gen_loss = gen_loss.mean(dim=[1, 2, 3]) # [BS]

                    # Apply Mask: Only count loss for Gen samples
                    masked_gen_loss = (gen_loss * loss_mask[:, 1]).mean()#loss_mask[:, 1].sum()=tensor(71., device='cuda:0', dtype=torch.bfloat16)

                    # =========================================================
                    # [NEW] Auxiliary Localization Loss (Consistency Check)
                    # =========================================================
                    # 仅当训练生成任务，且 actions (GT Pose) 存在时计算
                    # 并且为了显存安全，可能只对部分样本计算，或者需要 gradient checkpointing
                    if getattr(self.config, 'is_loc_aux_loss', False) and actions is not None:
                        # actions [BS, 1, 5] 即使是 Gen 任务，Dataset 也应该把 pose 传进来
                        masked_loc_aux_loss = self.forward_for_aux_loc_loss(
                            sigmas,
                            noisy_latents,
                            noise_pred,
                            aux_loc_input_ids,
                            aux_loc_labels,
                            und_image,
                            gen_image,
                            grid_thw,
                            i_s_pos,
                            image_sizes,
                            task_id,
                            return_dict,
                            actions,
                            loss_mask
                        )
                    else:
                        masked_loc_aux_loss = torch.nn.MSELoss()(hidden_states, torch.clone(hidden_states.detach())).to(torch.float32)

            else:
                masked_gen_loss = torch.nn.MSELoss()(hidden_states, torch.clone(hidden_states.detach())).to(torch.float32)
                masked_loc_aux_loss = torch.nn.MSELoss()(hidden_states, torch.clone(hidden_states.detach())).to(torch.float32)

            # ==========================================
            # Branch 2: LOCALIZATION (Flow Matching Path)
            # ==========================================
            if loss_mask[:, 0].sum() > 0: # If any sample needs Localization
                # actions: [BS, 1, 5]
                actions = actions#torch.Size([128, 5])
                # # TODO 这里也一样！如果使用[is_loc_task]过滤actions和其他中间tensor后再进行embed_action_suffix和action_dit，能节约显存，但是由于[is_loc_task]的数目不一定恰好等于2的次方，所以可能会影响cuda加速运算。除非数据集collactor手动设置loc:gen=64:64。
                # 1. Flow Matching Setup
                # Sample Noise & Time
                noise = self.sample_noise(actions.shape, actions.device)#torch.Size([128, 5])
                time = self.sample_time(actions.shape[0], actions.device)#torch.Size([128])

                # Interpolate: x_t = t * noise + (1-t) * x_1 (Actions)
                time_expanded = time[:, None, None].to(actions.dtype) # [BS, 1, 1] #torch.Size([128, 1, 1])
                x_t = time_expanded * noise + (1 - time_expanded) * actions # 带噪声的中间向量x_t #torch.Size([128, 1, 5])

                u_t = noise - actions # 需要模型预测的 Velocity Target 速度场 #torch.Size([128, 5])

                # 2. Prepare Inputs for Action Connector
                # We treat LLM hidden_states as "Prefix" (Context)
                # Embed the suffix (Noisy Action + Time)
                suffix_emb, adarms_cond = self.embed_action_suffix(
                    x_t, #torch.Size([128, 1, 5])
                    time, #torch.Size([128])
                    llm_hidden_size=self.model.config.text_config.hidden_size,
                    device=actions.device,
                    dtype=hidden_states.dtype
                ) # suffix_emb: [BS, 1, Hidden] #torch.Size([128, 1, 896])

                # 防止；anguage_model输出的last_hidden_state出现max=266，min=-256，而导致梯度nan
                # hidden_states = self.get_model().action_dit_norm(hidden_states)
                # action_emb = self.get_model().action_norm(action_emb)

                # scaler = self.model.config.text_config.hidden_size ** 0.5
                # suffix_emb = suffix_emb * scaler

                # 3. Concatenate & Forward Action Connector
                # Context (LLM Output) + Suffix (Action)
                # We need to construct attention mask so Suffix sees Context, but standard Causal mask is fine usually

                # Concat Embeddings
                # Scatter 填充 # “右移填空” (Right Shift & Fill) 操作 # 实现"Left Padding" 或 "Packing"的效果
                bs, seq_len, hidden_dim = hidden_states.shape
                valid_lens = attention_mask.sum(dim=1).long()

                # action_dit_inputs = torch.cat([hidden_states, suffix_emb], dim=1) # [BS, Seq+1, Hidden] #hidden_states=torch.Size([128, 707, 896]) #suffix_emb=torch.Size([128, 1, 896]) #action_dit_inputs #torch.Size([128, 708, 896])
                action_dit_inputs = torch.cat([hidden_states, torch.zeros_like(suffix_emb)], dim=1)
                target_indices = valid_lens.view(-1, 1, 1).expand(-1, 1, hidden_dim)#把 suffix_emb 放到 valid_lens 的位置 # 构造索引：我们需要修改的位置是 (b, valid_lens[b]) # view(-1, 1, 1) 是为了广播到 hidden_dim
                action_dit_inputs = action_dit_inputs.scatter(1, target_indices, suffix_emb)

                # Extend Masks
                # 1 for Action token (visible)
                # action_mask = torch.ones((bsz, 1, 1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                # action_mask = torch.ones((bsz, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                # action_dit_att_mask = torch.cat([attention_mask, action_mask], dim=1) #torch.Size([128, 708])
                action_dit_att_mask = torch.cat([attention_mask, torch.zeros((bs, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
                mask_indices = valid_lens.view(-1, 1)
                action_dit_att_mask = action_dit_att_mask.scatter(1, mask_indices, 1)

                # Position IDs
                # action_pos_id = position_ids.max(dim=1)[0].unsqueeze(1) + 1
                # action_dit_pos_ids = torch.cat([position_ids, action_pos_id], dim=1) #position_ids=torch.Size([128, 707]) #action_pos_id=torch.Size([128, 1])
                action_dit_pos_ids = torch.cat([position_ids, torch.zeros((bs, 1), device=position_ids.device, dtype=position_ids.dtype)], dim=1)
                # Action 的 Pos ID 应该是上一个 token 的 pos + 1，或者直接就是 valid_lens (如果从0开始)
                # 假设你的 position_ids 在 padding 处是 0 或其他，我们这里显式计算一下 action 的 pos
                action_pos_ids = valid_lens.view(-1, 1) # Action 的位置索引就是它的序列位置
                action_dit_pos_ids = action_dit_pos_ids.scatter(1, mask_indices, action_pos_ids)

                #### TODO
                # 这里action_dit_inputs和action_dit_pos_ids有一个风险点，为了将suffix_emb和action_pos_ids拼接到正确的位置，我们先使用zeros填充到正确的seq长度，然后再使用scatter找到正确位置valid_lens填充。而这样做会导致原先padding位置的tensor被zeros替代，虽然action_dit_att_mask不受影响且会忽略padding位置的tensor的梯度计算，但还是有风险(剧烈数值波动、某些bf16计算、特定FlashAttention算子等)。
                # 还有一个更糟糕的情况，如果手动将attention_mask的padding位置的position_id正确地替代到action_dit_pos_ids上是可以实现的。但是将hidden_states在padding位置的embed替代到正确的action_dit_inputs的位置是不可能的，因为hidden_states每个位置的embed是由所有位置的embed计算得到的。
                # 综上所述，我打算先仿照gen任务dit，将hidden_states作为encoder_hidden_states输入action_dit的UniLIP方式。以此来替代hidden_states拼接suffix_emb作为inputs_embeds输入action_dit的Pi05方式。
                # 但是gen任务的dit是个sana，支持timestep参数的输入；action_dit是个internvl.language_model需要模仿Pi05的gemma_300m将附带timestep信息的adarms_cond传入模型，且修改模型的norm layer已适配adarms_cond。
                # 第二次综上所述，先忽略zeros填充的风险点，跑通模型。后续再修改action_dit的代码来适配adarms_cond
                #### TODO

                # Forward Action Connector (InternVL Slice)
                # Reuse bidr mask logic or standard causal. Since it's InternVL, it expects eager/causal usually.
                # For simplicity, we assume bidr mask logic handles the sequence extension as default.
                # 注意在OpenPi0.5中使用的gemma_expert_model还会接受adarms_cond(一个跟timestep有关的embedding)作为输入
                if getattr(self.config, 'is_action_dit_dense_timestep', False):
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
                    action_hidden = action_outputs.hidden_states[-1][:, -1:, :] # [BS, 1, Hidden] #torch.Size([128, 1, 896])

                #### TODO
                # # 第二次综上所述，先忽略zeros填充的风险点，跑通模型。后续再修改action_dit的代码来适配adarms_cond
                # action_hidden = self.model.action_dit(
                #     suffix_emb, #torch.Size([128, 32, 16, 16])
                #     timestep=time, #128
                #     encoder_hidden_states=hidden_states, # [BS, Seq, C] ##torch.Size([128, 707, 2304])
                #     encoder_attention_mask=attention_mask, #torch.Size([128, 707])
                #     return_dict=False
                # )[0] #[BS, 1, Hidden]

                # 4. Final Projection (Velocity Prediction)
                v_t_pred = self.get_model().action_out_proj(action_hidden) # [BS, 1, 5] #torch.Size([128, 1, 5])

                # 5. Calculate Loss (MSE)
                loc_loss = F.mse_loss(v_t_pred.float(), u_t.float(), reduction="none") #torch.Size([128, 1, 5])
                loc_loss = loc_loss.mean(dim=[1, 2]) # [BS] #torch.Size([128])

                # Apply Mask: Only count loss for Loc samples
                masked_loc_loss = (loc_loss * loss_mask[:, 0]).mean()#loss_mask[:, 0].sum()=tensor(57., device='cuda:0', dtype=torch.bfloat16)
            else:
                masked_loc_loss = torch.nn.MSELoss()(hidden_states, torch.clone(hidden_states.detach())).to(torch.float32)

        alpha_loc_aux_loss = torch.tensor(self.model.config.alpha_loc_aux_loss).to(torch.float32)
        total_loss = masked_gen_loss + masked_loc_loss + masked_loc_aux_loss * alpha_loc_aux_loss
        logging.info(f"total_loss: {total_loss.detach().cpu().numpy().item():6f}, masked_loc_loss: {masked_loc_loss.detach().cpu().numpy().item():6f}, masked_gen_loss: {masked_gen_loss.detach().cpu().numpy().item():6f}, masked_loc_aux_loss: {masked_loc_aux_loss.detach().cpu().numpy().item():6f}, alpha_loc_aux_loss: {alpha_loc_aux_loss.detach().cpu().numpy().item():6f}")

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=None, # Not used
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_for_aux_loc_loss(
        self,
        sigmas,
        noisy_latents,
        noise_pred,
        aux_loc_input_ids,
        aux_loc_labels,
        und_image_map,
        gen_image,
        grid_thw,
        i_s_pos,
        image_sizes,
        task_id,
        return_dict,
        actions,
        loss_mask
    ):
        # 1. Estimate x_0 (Clean Latent) from current prediction
        # Flow Matching (Euler): x_t = (1-t)x_0 + t*x_1; v = x_1 - x_0
        # => x_0 = x_t - t * v (approx)
        # 注意: 这里的 sigmas 对应 t。target 对应 v (如果 noise_scheduler 是 rectified flow)
        # 如果是 standard diffusion, 公式不同。这里假设是 Rectified Flow (Sana Default)
        t_broadcast = sigmas
        pred_latents_x0 = noisy_latents - t_broadcast * noise_pred # \hat{x}_0 (Latent Space)

        # 2. VAE Decode (Latent -> Pixel)
        # [Gradient Flow] 梯度需要流经 pred_latents_x0 -> noise_pred -> DiT
        # 因此这里不能用 no_grad
        # scale back: UniLIP Factor
        pred_latents_scaled = pred_latents_x0 / self.model.config.unilip_factor
        # VAE Decode is heavy! Use with caution.
        pred_pixels = self.model.vae_decoder.vae_decode(pred_latents_scaled) # [BS, 3, H, W] (-1~1)

        # 3. Process for Vision Encoder (SigLIP)
        # SigLIP expects [0, 1] and specific normalization
        # pred_pixels is [-1, 1], convert to [0, 1]
        pred_pixels_norm = (pred_pixels + 1.0) / 2.0
        # Resize to Vision Encoder size (e.g. 448) if needed
        # VAE output is usually 512 or 1024. SigLIP is 448.
        if pred_pixels_norm.shape[-1] != 448:
            pred_pixels_norm = F.interpolate(pred_pixels_norm, size=(448, 448), mode='bilinear', align_corners=False)
        # SigLIP Normalization (Mean/Std) - Approximate or use processor values
        # mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5] for simplicity/speed in training loop
        # Or use self.image_processor logic
        pred_pixels_input = (pred_pixels_norm - 0.5) / 0.5

        # 4. Forward Localization Branch with Generated Image
        # [Important] Freeze Loc Branch weights to avoid updating them with noisy gradients
        # We only want gradients to flow back to the Image (pred_pixels_input)

        # 构造 Loc 分支输入
        # Und Image = Generated Fps Image (pred_pixels_input)
        # Aux Image = Original Map (und_image passed in forward, which is actually map for Gen task)
        # Prompt = Loc Prompt (aux_loc_input_ids)

        # 获取 Gen 任务原本的 Map 输入 (在 prepare_inputs 里它是 und_image)
        combined_und_images = torch.cat([pred_pixels_input, und_image_map], dim=0)
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
            torch.zeros_like(task_id),
        )
        und_img_idx = combined_img_idx[:combined_img_idx.size(0)//2, ...] #und_img_idx,sum()=32768 #32768/256=128.0
        aux_img_idx = combined_img_idx[combined_img_idx.size(0)//2:, ...]#aux_img_idx.sum()=tensor(14592, device='cuda:0') #14592/256=57
        und_image_embeds = combined_image_embeds[:combined_image_embeds.size(0)//2, ...]#torch.Size([128, 256, 896])
        aux_image_embeds = combined_image_embeds[combined_image_embeds.size(0)//2:, ...]#torch.Size([128, 256, 896])

        # --- B. Main LLM Forward (Understanding) ---
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

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

        # Re-fill und_image embeddings (Skip Connection logic from UniLIP)
        if und_image_embeds is not None and und_img_idx is not None:
            hidden_states[und_img_idx] = und_image_embeds.to(hidden_states.device).flatten(0,1)

        is_loc_task = (task_id == 0)#is_loc_task.shape=torch.Size([128])#is_loc_task.sum()=tensor(57, device='cuda:0')
        if aux_image_embeds is not None and und_img_idx is not None: #aux_img_idx.sum()/128 = 57
            hidden_states[aux_img_idx] = aux_image_embeds[is_loc_task].to(hidden_states.device).flatten(0,1)#hidden_states[aux_img_idx].shape=torch.Size([14592, 896]) #aux_image_embeds[is_loc_task].shape=torch.Size([57, 256, 896])

        # 5. Original LOCALIZATION Branch (Flow Matching Path)
        actions = actions#torch.Size([128, 5])
        noise = self.sample_noise(actions.shape, actions.device)#torch.Size([128, 5])
        time = self.sample_time(actions.shape[0], actions.device)#torch.Size([128])
        time_expanded = time[:, None, None].to(actions.dtype)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 临时冻结 Action Dit
        self.model.action_dit.requires_grad_(False)
        self.model.action_in_proj.requires_grad_(False)
        self.model.action_out_proj.requires_grad_(False)
        self.model.time_mlp_in.requires_grad_(False)
        self.model.time_mlp_out.requires_grad_(False)
        self.model.action_dit.eval()
        self.model.action_in_proj.eval()
        self.model.action_out_proj.eval()
        self.model.time_mlp_in.eval()
        self.model.time_mlp_out.eval()
        with torch.no_grad():
            suffix_emb, adarms_cond = self.embed_action_suffix(
                    x_t, #torch.Size([128, 1, 5])
                    time, #torch.Size([128])
                    llm_hidden_size=self.model.config.text_config.hidden_size,
                    device=actions.device,
                    dtype=hidden_states.dtype
                )
            bs, seq_len, hidden_dim = hidden_states.shape
            valid_lens = attention_mask.sum(dim=1).long()

            action_dit_inputs = torch.cat([hidden_states, torch.zeros_like(suffix_emb)], dim=1)
            target_indices = valid_lens.view(-1, 1, 1).expand(-1, 1, hidden_dim)#把
            action_dit_inputs = action_dit_inputs.scatter(1, target_indices, suffix_emb)

            action_dit_att_mask = torch.cat([attention_mask, torch.zeros((bs, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
            mask_indices = valid_lens.view(-1, 1)
            action_dit_att_mask = action_dit_att_mask.scatter(1, mask_indices, 1)

            action_dit_pos_ids = torch.cat([position_ids, torch.zeros((bs, 1), device=position_ids.device, dtype=position_ids.dtype)], dim=1)
            action_pos_ids = valid_lens.view(-1, 1) # Action 的位置索引就是它的序列位置
            action_dit_pos_ids = action_dit_pos_ids.scatter(1, mask_indices, action_pos_ids)

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
                action_hidden = action_outputs.hidden_states[-1][:, -1:, :] # [BS, 1, Hidden] #torch.Size([128, 1, 896])

            v_t_pred = self.get_model().action_out_proj(action_hidden) # [BS, 1, 5] #torch.Size([128, 1, 5])
            loc_loss = F.mse_loss(v_t_pred.float(), u_t.float(), reduction="none") #torch.Size([128, 1, 5])
            loc_loss = loc_loss.mean(dim=[1, 2]) # [BS] #torch.Size([128])

            # 加权：只在 t 小的时候 (生成接近完成) 计算 Loss
            # sigmas 越大噪声越大。我们希望 sigma 小的时候权重高。
            weight = (1.0 - sigmas).clamp(min=0)
            loc_loss = (loc_loss * weight.squeeze())

            # Apply Mask: Only count aux loc loss for Gen samples
            masked_loc_loss = (loc_loss * loss_mask[:, 1]).mean()

        # 临时冻结 Action Dit
        self.model.action_dit.requires_grad_(True)
        self.model.action_in_proj.requires_grad_(True)
        self.model.action_out_proj.requires_grad_(True)
        self.model.time_mlp_in.requires_grad_(True)
        self.model.time_mlp_out.requires_grad_(True)
        self.model.action_dit.train()
        self.model.action_in_proj.train()
        self.model.action_out_proj.train()
        self.model.time_mlp_in.train()
        self.model.time_mlp_out.train()

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
        text: List[str],
        tokenizer: AutoTokenizer,
        und_images: Optional[torch.Tensor] = None,
        aux_images: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        generator: Optional[torch.Generator] = None
    ):
        """
        Run inference for Localization task using Flow Matching Euler Solver.
        Adapts the logic from `forward` (Right-Shift & Fill) to ensure consistency.
        """
        # 1. Tokenize & Prepare Inputs
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        device = self.get_model().device
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)  # B x N

        # 2. Get Vision Features
        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy

        und_image_embeds = None
        if und_images is not None:
            und_image_embeds = self.model.get_image_features(
                pixel_values=und_images.to(device, dtype=self.get_model().vision_tower.dtype),
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        aux_image_embeds = None
        if aux_images is not None:
            aux_image_embeds = self.model.get_image_features(
                pixel_values=aux_images.to(device, dtype=self.get_model().vision_tower.dtype),
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        # 3. Embed Text & Replace Image Tokens
        text_embeds = self.get_model().language_model.embed_tokens(input_ids)
        und_image_idx, aux_image_idx = split_image_tokens(input_ids, IMAGE_TOKEN_IDX)

        if und_images is not None and und_image_idx.any():
            # Broadcast embeddings to batch size if needed (e.g. 1 image for all prompts)
            # Assuming standard [BS, C, H, W] input for simplicity based on collator
            # If batch sizes match, no repeat needed. If single image for batch, repeat.
            if und_image_embeds.shape[0] == 1 and text_embeds.shape[0] > 1:
                 und_image_embeds = und_image_embeds.repeat(text_embeds.shape[0], 1, 1)
            text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device).flatten(0,1)

        if aux_images is not None and aux_image_idx.any():
            if aux_image_embeds.shape[0] == 1 and text_embeds.shape[0] > 1:
                 aux_image_embeds = aux_image_embeds.repeat(text_embeds.shape[0], 1, 1)
            text_embeds[aux_image_idx] = aux_image_embeds.to(text_embeds.device).flatten(0,1)

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
             hidden_states[und_image_idx] = und_image_embeds.to(hidden_states.device).flatten(0,1)
        if aux_image_embeds is not None and aux_image_idx.any():
             hidden_states[aux_image_idx] = aux_image_embeds.to(hidden_states.device).flatten(0,1)

        # 6. Initialize Flow Matching Loop
        bsize = input_ids.shape[0]
        action_dim = self.model.config.action_dim
        # # Sample initial noise x_1
        # x_t = randn_tensor(
        #     (bsize, 1, action_dim),
        #     generator=generator,
        #     device=device,
        #     dtype=hidden_states.dtype
        # )
        noise = self.sample_noise((bsize, 1, action_dim), device=device)

        dt = -1.0 / num_steps
        # dt tensor for calculation
        dt = torch.tensor(dt, device=device, dtype=hidden_states.dtype)

        x_t = noise
        t = torch.tensor(1.0, device=device, dtype=hidden_states.dtype)

        # Calculate Valid Lengths for Right-Shift Insertion
        # [BS, 1]
        valid_lens = attention_mask.sum(dim=1, keepdim=True).long()
        bs, seq_len, hidden_dim = hidden_states.shape

        # 7. Euler Solver Loop
        # Stop at t=0 (or close to it, Pi0 uses -dt/2 for safety)
        while t >= -dt / 2:

            # A. Embed Suffix (Noisy Action + Time)
            expanded_time = t.expand(bsize)
            suffix_emb, adarms_cond = self.embed_action_suffix(
                x_t,
                expanded_time,
                llm_hidden_size=self.model.config.text_config.hidden_size,
                device=device,
                dtype=hidden_states.dtype
            )

            # B. Construct Action Inputs (Right Shift & Fill Strategy)
            # Reusing the robust logic from forward pass to avoid NaN

            # 1. Inputs: Concat Hidden + Last Token (Safe Padding)
            last_token_states = hidden_states[:, -1:, :]
            extended_inputs = torch.cat([hidden_states, last_token_states], dim=1)

            # Scatter Suffix to valid positions
            scatter_indices = valid_lens.unsqueeze(-1).expand(-1, -1, hidden_dim)
            action_dit_inputs = extended_inputs.scatter(1, scatter_indices, suffix_emb)

            # 2. Mask: Extend and Set True at Action Position
            extended_mask = torch.cat([
                attention_mask,
                torch.zeros((bs, 1), device=device, dtype=attention_mask.dtype)
            ], dim=1)
            mask_indices = valid_lens.view(-1, 1)
            action_dit_att_mask = extended_mask.scatter(1, mask_indices, 1)

            # 3. Position IDs: Extend and Set to Valid Length Index
            extended_pos_ids = torch.cat([
                position_ids,
                torch.zeros((bs, 1), device=device, dtype=position_ids.dtype)
            ], dim=1)
            action_pos_ids = valid_lens.view(-1, 1)
            action_dit_pos_ids = extended_pos_ids.scatter(1, mask_indices, action_pos_ids)

            # # Safety Clamp
            # max_seq_len = action_dit_inputs.shape[1]
            # action_dit_pos_ids = action_dit_pos_ids.clamp(max=max_seq_len - 1)

            # C. Forward Action DiT
            if getattr(self.config, 'is_action_dit_dense_timestep', False):
                # Use custom forward loop with AdaRMS
                action_outputs = self.action_dit_forward_with_adarmscond(
                    hidden_states=action_dit_inputs,
                    attention_mask=action_dit_att_mask,
                    position_ids=action_dit_pos_ids,
                    adarms_cond=adarms_cond
                )
                # Note: custom forward returns hidden_states directly
                all_hidden_states = action_outputs
            else:
                # Use standard model forward
                action_outputs = self.model.action_dit(
                    inputs_embeds=action_dit_inputs,
                    attention_mask=action_dit_att_mask,
                    position_ids=action_dit_pos_ids,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False
                )
                all_hidden_states = action_outputs.hidden_states[-1]

            # D. Extract Action Token Output
            # Gather from the same indices we scattered to
            gather_indices = valid_lens.unsqueeze(-1).expand(-1, -1, hidden_dim)
            action_feat = all_hidden_states.gather(1, gather_indices) # [BS, 1, Hidden]

            # E. Predict Velocity & Step
            v_t = self.get_model().action_out_proj(action_feat)

            x_t = x_t + dt * v_t
            t += dt

        return x_t # Final Denoised Action [BS, 1, 5]



AutoConfig.register("unified_unilip", Unified_UniLIP_InternVLConfig)
AutoModelForCausalLM.register(Unified_UniLIP_InternVLConfig, Unified_UniLIP_InternVLForCausalLM)