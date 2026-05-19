# CSGO Experiment Guide

本文件用于整理 `csgo_configs` 下当前有效的实验脉络、每个实验的目的，以及后续 REPA / loc-aware REPA 实验的规划矩阵，便于后续在新线程中继续补配置、改代码和安排 ablation。

## Scope

当前重点讨论的是 `dust2` 上的 unified 训练路线，核心目标是回答三类问题：

1. 定位辅助生成是否有效。
2. 这种提升是否来自 `aux_loc_loss`，还是仅仅来自 joint multitask。
3. 在 unified 框架下，进一步引入表征对齐类监督时，哪种 teacher / 哪种表征挂点更有效。

## UniLIP Optimization and LoRA Defaults

本节记录当前 CS:GO unified 训练的学习率和 LoRA 约定。关键前提：原始 UniLIP 的 `learning_rate=1e-4` 是在冻结主 `language_model`、`vision_tower`、`multi_modal_projector`、VAE/decoder 后，用来训练 generation connector 与 DiT 的学习率；不要把它直接理解成全量 LLM / ViT 微调学习率。

### Original UniLIP Finetune Baseline

原始 UniLIP 中对外的 `UniLIP-3B` 在代码脚本里对应 `2b` 路线：`InternVL3-2B` + `SANA-1.6B`，README 中按总生成系统规模称为 3B。`UniLIP-1B` 和 `UniLIP-3B` 在 stage2 pretraining 与 stage3 SFT 的微调开关相同，只是基座与 DiT 规模不同。

| 模型 | 脚本 | 基座 / DiT | 微调阶段可学习模块 | 冻结模块 | 原始 LR 配置 |
|---|---|---|---|---|---|
| `UniLIP-1B` | `run_unilip_1b_stage2.sh`, `run_unilip_1b_stage3.sh` | `InternVL3-1B(-hf)` + `SANA-0.6B` | `dit`, `llm_connector`, `projector`, `latent_queries` | 主 `language_model`, `vision_tower`, `multi_modal_projector`, `vae_decoder` | `learning_rate=1e-4`, `min_lr=1e-5`, `weight_decay=0`, `warmup_ratio=0.003`, `cosine_with_min_lr` |
| `UniLIP-3B` | `run_unilip_2b_stage2.sh`, `run_unilip_2b_stage3.sh` | `InternVL3-2B(-hf)` + `SANA-1.6B` | `dit`, `llm_connector`, `projector`, `latent_queries` | 主 `language_model`, `vision_tower`, `multi_modal_projector`, `vae_decoder` | 同 `UniLIP-1B` |
| stage1 connector training | `run_unilip_1b_stage1.sh`, `run_unilip_2b_stage1.sh` | 同各自模型 | `llm_connector`, `projector`, `latent_queries` | `dit`, 主 `language_model`, `vision_tower`, `multi_modal_projector`, VAE/decoder | 同 `learning_rate=1e-4` |

因此 CS:GO 继续训练的默认基线应保持：generation `dit` 和 connector 类模块可以从 `1e-4` 起；主 LLM / ViT 若仍冻结，不需要单独 LR；若解冻或 LoRA 化，必须独立分组。

### CS:GO Unified LR Groups

| 模块 | Full / Trainable LR | LoRA LR | LoRA 配置 | 备注 |
|---|---:|---:|---|---|
| 主 `language_model` | freeze；解冻起点 `1e-5` | `1e-4` | `r=16`, `alpha=32`, `dropout=0.05`; target: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | 原始 UniLIP stage2/3 冻结主 LLM；只有 `llm_train_mode=full/lora` 时才训练 |
| shared LLM tail | `1e-5 ~ 2e-5` | `1e-4` | `shared_llm_tail_lora_r=16`, `alpha=32`, `dropout=0.05`; target 同 LLM | 用于 `shared_tail_full/shared_tail_lora`，不要同时打开 whole-LLM 训练 |
| `vision_tower` | freeze；解冻 `1e-6 ~ 3e-6` | `5e-5` | `r=8~16`, `alpha=2r`, `dropout=0.05`; target: `qkv,proj,fc1,fc2` | 原始 UniLIP 默认冻结；若解冻必须低 LR，避免破坏视觉表征 |
| `multi_modal_projector` | freeze；仅 `train_mm_projector_only` 时 `1e-4` | full train `1e-4` | 不建议 LoRA | 原始 UniLIP stage2/3 冻结；CS:GO 如需重新对齐再打开 |
| `llm_connector` | `1e-4` | `1e-4` | `r=8` when global `lora_r=16`, `alpha=32`, `dropout=0.05`; target 同 LLM | 原始 UniLIP 可学习模块；当前实现对 connector LoRA 使用 `lora_r // 2` |
| `projector` / `latent_queries` | `1e-4` | full train `1e-4` | 不建议 LoRA | 原始 UniLIP 可学习模块；connector 到生成分支的对齐层 |
| generation `dit` | `1e-4`；不稳时 `5e-5` | `5e-5 ~ 1e-4` | `r=16`, `alpha=16~32`, `dropout=0.05`; target: `to_q,to_k,to_v,to_out.0,linear_1,linear_2` | 原始 UniLIP stage2/3 full train；CS:GO joint loss 抖动时先降到 `5e-5` |
| `action_dit` | `5e-5 ~ 1e-4` | `5e-5 ~ 1e-4` | `r=16`, `alpha=16~32`, `dropout=0.05`; target: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Pi0.5 初始化建议 `5e-5`；新训 loc 分支可试 `1e-4` |
| `action_dit_connector` | `1e-4` | full train `1e-4` | 不建议 LoRA | 小连接层，保持 full train |
| `action_dit_norm` | `5e-4` | full train `5e-4` | 不建议 LoRA | norm / head 类参数量小 |
| `action_dit_projector` | `5e-4` | full train `5e-4` | 不建议 LoRA | 当前项目已有 `action_dit_projector_lr` |
| `action_in_proj` / `action_out_proj` | `1e-4 ~ 5e-4` | full train `1e-4 ~ 5e-4` | 不建议 LoRA | action 输入输出头 |
| `time_mlp_in` / `time_mlp_out` | `1e-4 ~ 5e-4` | full train `1e-4 ~ 5e-4` | 不建议 LoRA | timestep 小 MLP |
| `regression_loc_head` | `5e-4` | full train `5e-4` | 不建议 LoRA | 定位回归头 |
| `cross_view_fusion` | `5e-4` | full train `5e-4` | 不建议 LoRA | `use_vit_regression_head=True` |
| `vit_loc_fusion` | `5e-4` | full train `5e-4` | 不建议 LoRA | `use_codex_vit_regression_head=True` |
| `loc_learnable_query` | `5e-4` | full train `5e-4` | 不建议 LoRA | 当前项目已有 `loc_learnable_query_lr` |
| VAE / decoder | freeze；必须训则 `1e-6` | freeze | 不建议 LoRA | 原始 UniLIP stage2/3 冻结，不建议和主干一起训练 |

推荐 full finetune 起点：

```yaml
llm_train_mode: full
llm_lr: 1.0e-5
learning_rate: 1.0e-4
mm_projector_lr: 1.0e-4
action_dit_lr: 5.0e-5
action_dit_projector_lr: 5.0e-4
weight_decay: 0.0
warmup_ratio: 0.003
lr_scheduler_type: cosine_with_min_lr
lr_scheduler_kwargs:
  min_lr: 1.0e-5
```

推荐 LoRA 起点：

```yaml
is_lora: True
llm_train_mode: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
llm_lora_lr: 1.0e-4
learning_rate: 1.0e-4
mm_projector_lr: 1.0e-4
action_dit_lr: 5.0e-5
action_dit_projector_lr: 5.0e-4
weight_decay: 0.0
warmup_ratio: 0.003
lr_scheduler_type: cosine_with_min_lr
lr_scheduler_kwargs:
  min_lr: 1.0e-5
```

使用原则：

- 原始 UniLIP 风格继续训练：保持 `llm_train_mode=frozen`, `fix_vit=True`, `fix_connect=False`, `fix_dit=False`，让 `learning_rate=1e-4` 只覆盖 generation `dit`、`llm_connector`、`projector`、`latent_queries` 以及 CS:GO 新增可学习模块。
- Full finetune 时必须单独给主 `language_model`、`vision_tower`、shared tail、`action_dit`、heads/projectors 分组；不要让 LLM 或 ViT 默认落入 `learning_rate=1e-4`。
- 当前 `inject_lora_to_sub_module` 会在 `is_lora=True` 时按开关给 `vision_tower`、主 `language_model`、`llm_connector`、generation `dit`、`action_dit` 注入 LoRA，并手动保持 heads/projectors full train；原始 UniLIP stage2/3 本身不使用 LoRA。
- LoRA 主要节省显存和 optimizer state，不保证单 step 更快；forward 仍要跑完整 base，并额外增加 LoRA adapter matmul。
- 如果 aux-loc / action loss 抖动，优先把 `action_dit_lr` 或 generation `dit` 的有效 LR 降到 `5e-5`，或把 `lora_alpha` 从 `32` 降到 `16`；不要一开始降低 rank。
- 如果显存允许且数据量足够，`r=32, alpha=64` 可作为第二组 LoRA 实验；默认仍从 `r=16, alpha=32` 起。注意 `llm_connector` 和 `vision_tower` 在当前实现中使用 `lora_r // 2`。
- `vision_tower` 默认 freeze。若必须训练，需要独立低 LR 参数组，推荐 `1e-6 ~ 3e-6`。

参考依据：

- 原始 UniLIP stage2 / stage3 脚本：`scripts/run_unilip_1b_stage2.sh`, `scripts/run_unilip_1b_stage3.sh`, `scripts/run_unilip_2b_stage2.sh`, `scripts/run_unilip_2b_stage3.sh` 均使用 `fix_llm=True`, `fix_dit=False`, `fix_connect=False`, `learning_rate=1e-4`, `weight_decay=0`, `warmup_ratio=0.003`, `cosine_with_min_lr(min_lr=1e-5)`。
- 原始 UniLIP 可学习模块由 `unilip/model/unilip_internvl.py::initialize_vision_modules` 和 `unilip/train/train_stage{2,3}.py` 的 `fix_vit/fix_llm` 逻辑共同决定：stage2/3 冻结主 LLM / ViT / projector / VAE，训练 `dit`、`llm_connector`、`projector`、`latent_queries`。
- InternVL3 finetune 文档：`InternVL3-1B` full / LoRA finetune 资源说明，并默认 freeze visual encoder。<https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html>
- OpenGVLab InternVL3-1B full finetune 脚本：`freeze_llm=False`, `freeze_mlp=False`, `freeze_backbone=True`, `learning_rate=1e-5`, `weight_decay=0.05`, `warmup_ratio=0.03`, cosine。<https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_1b_dynamic_res_2nd_finetune_full.sh>
- OpenGVLab InternVL LoRA 实现：vision LoRA target 为 `attn.qkv/attn.proj/mlp.fc1/mlp.fc2`，Qwen/LLaMA LLM target 为 `q/k/v/o + gate/down/up`，默认 `alpha=2*r`。<https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py>
- OpenGVLab InternVL2.5-1B LoRA 脚本：`use_llm_lora=16`, `learning_rate=4e-5`, `weight_decay=0.01`, `warmup_ratio=0.03`，可作为更保守 LoRA 起点参考。<https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora.sh>
- LoRA 原论文：冻结 base、训练低秩 adapter，可显著减少可训练参数和 optimizer/显存压力。<https://arxiv.org/abs/2106.09685>
- QLoRA / PEFT 实践：大模型 LoRA 常对所有 linear 层注入 adapter，以更接近 full finetune 能力。<https://arxiv.org/abs/2305.14314>, <https://huggingface.co/docs/peft/main/developer_guides/lora>
- LoRA+ / LoRA-GA：LoRA 学习率、初始化和 A/B 矩阵优化策略会显著影响收敛速度。<https://arxiv.org/abs/2402.12354>, <https://arxiv.org/abs/2407.05000>

## Existing Experiment Families

### Single-task teachers and warm-start sources

| 配置 | 目的 | 说明 |
|---|---|---|
| `exp14_gen` | 单任务生成 baseline / 生成 warm-start 来源 | 非 `dust2` 专用版本 |
| `exp14_loc` | 单任务定位 baseline / 定位 warm-start 来源 | 非 `dust2` 专用版本 |
| `exp14_dust2_gen` | `dust2` 单任务生成 teacher / warm-start 来源 | 可作为 domain-specific vision teacher 候选 |
| `exp14_dust2_loc` | `dust2` 单任务定位 teacher / warm-start 来源 | 当前 loc-aware REPA teacher 的主要来源 |
| `exp14_1_dust2_gen` | `dust2` 生成分支变体 | 保留作参考，不是当前主线 |

### Auxiliary localization line

| 配置 | 目的 | 说明 |
|---|---|---|
| `exp15` | 固定定位 teacher 辅助生成 | 只训练生成样本，`aux_loc_loss` 打开，内部 loc teacher 来自已有定位权重 |
| `exp15_1` | 联合 `loc/gen` 训练并使用 `aux_loc_loss` | moving teacher 版本 |
| `exp15_2` | `exp15` 的动态 `alpha_loc_aux` 版本 | 控制辅助定位损失前期更保守 |
| `exp15_3` | `exp15_1` 的动态 `alpha_loc_aux` 版本 | 当前 joint + aux 的重要参照 |

### External localization teacher line

| 配置 | 目的 | 说明 |
|---|---|---|
| `exp16` | 外部定位器辅助生成 | 验证外部 frozen locator 作为 teacher 是否有效 |
| `exp16_1` | `exp16` 对照组 | 关闭 `is_loc_aux_loss` |
| `exp16_2` | `exp16` 调参版 | 保留作参考 |
| `exp16_3` | `exp16` 调参版 | 保留作参考 |
| `exp16_4` | `exp16` 调参版 | 保留作参考 |

### Unified from UniLIP initialization

| 配置 | 目的 | 说明 |
|---|---|---|
| `exp17` | 从原始 `UniLIP-1B` 出发做 unified joint 训练 | 不加载任何 CS:GO warm-start |
| `exp17_1` | `exp17` 的动态 `alpha_loc_aux` 版本 | schedule: `[0,3000,4000,5000,10000] -> [0,1,2,5,10]` |
| `exp17_2` | `exp17_1` + 动态 `alpha_loc` / 新 loc lr grouping | 当前 unified 主线基线 |
| `exp17_2_dust2` | `exp17_2` 的 `dust2` 专用版 | 当前 `dust2` 主 baseline |
| `exp19_dust2` | `exp17_2_dust2` 的 `aux_loc` step-level periodic gate 版本 | 在保留 `alpha_loc_aux` schedule 的同时加入周期 gate，并在 gate 关闭时跳过 aux_loc 前向链路 |
| `exp20_dust2` | `exp17_2_dust2` 的 noisy-loc distribution matching 版本 | 对 batch 内随机 `30%` loc 子集做 latent-space matched noise，并仅对 noisy 子集覆盖 `1-sigma` loss 权重 |
| `exp21_dust2` | `exp17_2_dust2` 的 stochastic generalized EM-inspired aux-loc 已实现配置 | 对同一样本采样多个 `x0_hat` candidate，用 loc consistency 估计 latent responsibility，再加权更新生成侧 |
| `exp22_dust2` | `exp17_2_dust2` 的 uncertainty-weighted residual consistency aux-loc 已实现配置 | 对同一样本采样两个 `x0_hat` candidate，用 loc velocity residual disagreement 作为样本级 aux 权重 |
| `exp23_dust2` | `exp17_2_dust2` 的 combined EM-responsibility + uncertainty-gated aux-loc 已实现配置 | 同时使用 candidate-level responsibility 和 sample-level residual consistency 权重 |
| `exp24_dust2` | `exp17_7_dust2@8000` 的 224 输入 + 短 instruction + combined aux-loc 继续训练配置 | 保留 DINOv2 standard REPA，启用 `img_size=224`、`use_short_instruction=True` 和 `exp23` 的 combined aux-loc |
| `exp25_dust2` | `exp24_dust2` 的双向 auxiliary consistency 版本 | 新增 `aux_gen_loss`，与 `aux_loc_loss` 按 iteration 交替；`aux_loc` 更新生成侧，`aux_gen` 冻结生成侧并通过可微 pose token 让 gen loss 更新定位侧 |
| `exp26_dust2` | `exp17_2_2_dust2` 的 224 + 短 instruction + combined aux-loc 干净起点对照 | 从原始 `UniLIP-1B` 起训，不使用 warm-start / REPA / aux_gen，用于隔离分辨率、短 prompt 和 combined aux-loc 的影响 |
| `exp26_1_dust2` | `exp26_dust2` 的 low-noise-only aux-loc timestep 版本 | 去掉现有 `1-sigma` 连续权重，仅在 `sigma <= 0.35` 的生成低噪声 timestep 上计算 aux-loc，并用 active mean 归一 |
| `exp26_2_dust2` | `exp26_dust2` 的 exponential low-noise-biased aux-loc timestep 版本 | 用 `exp(-5*sigma)` 替代现有 `1-sigma` 连续权重，保留所有 timestep 但强烈偏向低噪声 |
| `exp27_dust2` | `exp26_dust2` 去掉 aux-loc 后加入 current-loc-head perceptual loss | 只用当前 loc head 的 action_dit_projector patch feature 对齐生成侧和定位侧特征；前 2000 step 关闭，之后 `alpha=0.1`；attention weighting 留给 `exp27_1_dust2` |
| `exp27_1_dust2` | `exp27_dust2` + attention-weighted current-loc-head perceptual loss | 最终固定方案：`teacher_gt + last layer + mean heads + mean_one + detach + loc_sampled`；用 wrapper 从 action_dit 最后一层取 action token 到 und patch tokens 的 attention，不用 hook |
| `exp27_2_dust2` | `exp27_dust2` 的浅层 vision_tower perceptual loss 对照 | 不使用 attention weighting，直接对齐 `pred_pixels_input` 和 `gen_image` 的 FPS vision patch features，`smooth_l1`，特征维度 `[B, 256, D_llm]` |
| `exp27_3_dust2` | `exp27_2_dust2` + teacher_gt action_dit attention weighting | perception feature 仍是 `vision_tower` FPS patch features；attention 权重复用 `teacher_gt + last layer + mean heads + mean_one + detach + loc_sampled` wrapper，不用 hook |
| `exp28_dust2` | `exp26_1_dust2` + `exp27_3_dust2` 的组合实验 | low-noise-only combined aux-loc 与 teacher_gt attention-weighted vision_tower perceptual alignment 同时开启，验证 pose-level 与 patch-feature-level 监督是否互补 |
| `exp17_3` | shared `multi_modal_projector` 联合训练 | 生成结果显著退化，说明 shared 位置不合适 |
| `exp17_4` | shared `language_model` tail 联合训练 | 用于替代 `exp17_3` 的更安全 shared 方案 |
| `exp17_4_dust2` | `exp17_4` 的 `dust2` 专用版 | `dust2` shared-tail baseline |
| `exp17_4_1_dust2` | `exp17_4_dust2` 的更深 shared-tail 版本 | `shared_llm_tail_num_layers=6` 的 full finetune 对照 |

### Selective warm-start unified line

| 配置 | 目的 | 说明 |
|---|---|---|
| `exp18` | base/loc 从 `exp14_loc`，gen 从 `exp14_gen` 的 joint warm-start | 非 `dust2` 专用 |
| `exp18_dust2` | `exp18` 的 `dust2` 专用版 | 用于比较 warm-start 与 scratch unified |
| `exp18_1_dust2` | `exp18_dust2` 的 `dust2` 调整版 | 改用 `exp14_1_dust2_gen@8000` 作为 gen init，并降低 loc lr |

## Dust2 Mainline: Current Anchor Experiments

以下实验是 `dust2` 上当前最重要的锚点，后续所有 REPA ablation 都应围绕它们展开：

| 配置 | 当前定位 | 作用 |
|---|---|---|
| `exp17_2_dust2` | 主 baseline | `gen + loc + aux_loc` |
| `exp19_dust2` | `aux_loc` 周期 gate baseline | `exp17_2_dust2 + step-level periodic gate for aux_loc` |
| `exp20_dust2` | noisy-loc matching baseline | `exp17_2_dust2 + latent-space noisy loc subset + per-sample (1-sigma) weighting` |
| `exp21_dust2` | EM-inspired aux-loc 已实现配置 | `exp17_2_dust2 + multi-sample x0_hat candidates + detached responsibility-weighted aux_loc` |
| `exp22_dust2` | uncertainty-weighted aux-loc 已实现配置 | `exp17_2_dust2 + two-sample x0_hat residual-disagreement sample weighting` |
| `exp23_dust2` | combined EM + uncertainty aux-loc 已实现配置 | `exp17_2_dust2 + candidate responsibility + residual-consistency sample gating` |
| `exp24_dust2` | 224 + short-instruction REPA/aux-loc warm-start | `exp17_7_dust2@8000 + img_size=224 + use_short_instruction=True + exp23 combined aux_loc` |
| `exp25_dust2` | 双向 alternating aux consistency | `exp24_dust2 + aux_gen_loss；even step 跑 aux_loc，odd step 跑 aux_gen` |
| `exp26_dust2` | 224 + short-instruction combined aux-loc scratch 对照 | `exp17_2_2_dust2 + exp23 combined aux_loc + img_size=224 + use_short_instruction=True`，原始 `UniLIP-1B` 起训 |
| `exp26_1_dust2` | low-noise-only combined aux-loc 对照 | `exp26_dust2 + aux_loc_timestep_weight_type=low_noise_only`，只保留 `sigma <= 0.35` 的 aux-loc candidate |
| `exp26_2_dust2` | exponential low-noise-biased combined aux-loc 对照 | `exp26_dust2 + aux_loc_timestep_weight_type=exp_sigma, aux_loc_exp_weight_lambda=5.0`，保留所有 candidate 但按 `exp(-5*sigma)` 降权 |
| `exp27_dust2` | current-loc-head perceptual alignment 对照 | `exp26_dust2 - aux_loc_loss + current loc feature smooth_l1`，仅验证 feature-level perceptual alignment |
| `exp27_1_dust2` | attention-weighted current-loc-head perceptual alignment | `exp27_dust2 + action_dit attention-weighted loc perception loss`，attention source 固定为 `teacher_gt + last + mean heads + mean_one + detach + loc_sampled` |
| `exp27_2_dust2` | shallow vision_tower perceptual alignment 对照 | `exp27_dust2` 的更浅对齐版本，`loc_perception_feature_source=vision_tower`，无 attention weighting |
| `exp27_3_dust2` | attention-weighted shallow vision_tower perceptual alignment | `exp27_2_dust2 + teacher_gt action_dit attention weighting`，权重方案固定为 `teacher_gt + last + mean heads + mean_one + detach + loc_sampled` |
| `exp28_dust2` | low-noise aux-loc + attention-weighted vision_tower perceptual alignment | `exp26_1_dust2 + exp27_3_dust2`，同时使用 low-noise-only combined aux-loc 和 teacher_gt action attention-weighted vision_tower feature alignment |
| `exp17_4_dust2` | shared-tail baseline | `exp17_2_dust2 + 2-layer shared LLM tail` |
| `exp17_4_1_dust2` | deeper shared-tail baseline | `exp17_4_dust2 + 6-layer shared LLM tail full finetune` |
| `exp17_5_dust2` | loc-aware REPA baseline | `exp17_2_dust2 + independent loc-aware REPA` |
| `exp17_6_dust2` | loc-aware REPA + shared tail | `exp17_5_dust2 + train_shared_llm_tail_only` |
| `exp17_6_1_dust2` | loc-aware REPA + deeper shared tail | `exp17_6_dust2` 的 6-layer shared tail full finetune 对照 |
| `exp17_6_2_dust2` | loc-aware REPA + 12-layer shared tail lora_only | 资源受限下的更深 shared tail 替代方案 |
| `exp17_7_dust2` | traditional REPA baseline | `exp17_2_dust2 + DINOv2 teacher + DiT layer-6 patch-wise REPA` |
| `exp17_8_dust2` | traditional REPA + aux_loc | `exp17_7_dust2 + aux_loc` |
| `exp18_dust2` | warm-start 对照 | 用于比较 warm-start joint 与 scratch joint |
| `exp18_1_dust2` | warm-start 调整版 | 比较不同 gen warm-start 来源和更保守 loc lr 的影响 |

### Aux-loc step gate support

当前代码已经支持对 `aux_loc_loss` 做 step-level periodic gate，适用于 `exp17_2_dust2` 这类 unified baseline 的变体实验：

- `effective_alpha_loc_aux(step) = scheduled_alpha_loc_aux(step) * periodic_gate(step)`
- 当当前 step 的 `effective_alpha_loc_aux == 0` 时：
  - 不进入 `forward_for_aux_loc_loss()`
  - 如果该 step 也没有 `loc_repa_loss`，则连 `pred_pixels_input` 的解码也一起跳过
- 这意味着 gate 关闭时不仅不更新生成分支的 aux 路径，也能节省显存和训练时间

当前可用配置键：

```yaml
is_loc_aux_step_gate: True
loc_aux_gate_cycle_steps: 300
loc_aux_gate_on_steps: 100
loc_aux_gate_start_step: 0
```

对应实验：

- `exp19_dust2`
  - 基于 `exp17_2_dust2`
  - 保留 `alpha_loc_aux_schedule_steps/values`
  - 再乘一个周期 gate
  - train/test 配置已补齐：
    - `csgo_configs/exp19_dust2.yaml`
    - `csgo_configs/test/exp19_dust2_gen.yaml`
    - `csgo_configs/test/exp19_dust2_gen_conti.yaml`
    - `csgo_configs/test/exp19_dust2_loc.yaml`

## Noisy Loc Matching Line

### Design summary

当前代码已经支持 `exp20_dust2` 这类 noisy-loc distribution matching 实验，目标是让定位分支在训练时看到一部分与 `loc_aux_loss` 中 `x0_hat` 更接近的数据分布：

- 基础配置：
  - 直接基于 `exp17_2_dust2`
  - 保持 `is_multi_task=True`
  - 保持 `is_multi_task_balanced=True`
- noisy loc 子集：
  - 在每个 batch 的 loc 样本内随机抽样
  - 当前默认比例为 `0.3`
- noisy 构造方式：
  - 先将 loc FPS 图像编码到与生成分支一致的 target latent 空间
  - 使用与生成分支一致的 timestep / sigma 采样逻辑加噪
  - 再 decode 回 pixel space，替换 loc 子集对应的干净图像
- loss 权重：
  - clean loc 样本保持权重 `1.0`
  - noisy loc 子集单独覆盖为 `1 - sigma`
- 当前实现范围：
  - 只支持 `noisy_loc_image_source=latent_space`
  - 只支持 `noisy_loc_sigma_sampling=gen_matched`
  - 只支持 `noisy_loc_weight_type=linear_1m_sigma`

### Implemented configs

- `exp20_dust2`
  - 基于 `exp17_2_dust2`
  - 新增：
    - `is_noisy_loc_loss=True`
    - `noisy_loc_ratio=0.3`
    - `noisy_loc_image_source=latent_space`
    - `noisy_loc_sigma_sampling=gen_matched`
    - `noisy_loc_weight_type=linear_1m_sigma`
  - train/test 配置已补齐：
    - `csgo_configs/exp20_dust2.yaml`
    - `csgo_configs/test/exp20_dust2_gen.yaml`
    - `csgo_configs/test/exp20_dust2_gen_conti.yaml`
    - `csgo_configs/test/exp20_dust2_loc.yaml`

### Current notes

- 该实现直接对 loc 图像单独编码 latent，不复用 `prepare_inputs_labels_for_multimodal()` 返回值，目的是避免在主 prepare 调用之前引入额外的 batch 对齐和中间状态耦合。
- 当前 loss 权重覆盖仅作用于 noisy loc 子集，clean loc 样本仍保持原始训练目标。
- 该实现面向 `exp17_2_dust2` 当前主路径，即 `use_pi05_action_dit=True` 的定位头路线。

### Future directions

- `x0_hat` matched noise：
  - 不再只做 schedule-level matching，而是尝试直接对齐 `loc_aux_loss` 实际使用的 `pred_latents_x0 -> vae.decode` 分布。
- noisy ratio schedule：
  - 将 `noisy_loc_ratio` 从固定值扩展为 schedule，先小比例 warmup，再逐步提高。
- loc weight normalization：
  - 比较当前直接乘 `1-sigma` 与按 noisy 子集重新归一化后的加权方式，区分“分布匹配”与“总 loc 强度变化”两个因素。
- clean/noisy consistency：
  - 对同一样本的 clean/noisy loc feature 或 action prediction 增加一致性约束，减少 noisy augmentation 带来的定位漂移。
- paired-gen latent reuse ablation：
  - 如果后续明确要进一步压缩开销，可评估是否直接复用 paired gen 样本的 latent/noise，而不是对 loc 图像单独再编码一遍。
- migrate best variant to `exp17_5_dust2`：
  - 如果 noisy-loc matching 在 `exp17_2_dust2` 上成立，再迁移到 loc-aware REPA 主线，检查其与 `loc_repa_loss` 是否互补。

## Stochastic Generalized EM-inspired Aux-Loc Line

### Design summary

`exp21_dust2` 把当前单个 `x0_hat` 的 `aux_loc_loss` 扩展成 multi-sample latent expectation 近似。它不是标准 closed-form EM，而是更接近 stochastic generalized EM：

- latent candidate:
  - 对同一个 gen 样本采样 `K` 个 `(timestep, noise)`
  - 分别得到 `K` 个 `x0_hat_k = vae.decode(pred_latents_x0_k)`
- E-like step:
  - 冻结 loc evaluator
  - 对每个 `x0_hat_k` 计算 `loc_loss_k`
  - 用 detached `loc_loss_k` 估计 candidate responsibility

```text
q_k = softmax(- stopgrad(loc_loss_k) / tau_candidate)
```

- M-like step:
  - 固定 `q_k`
  - 用加权 aux loss 更新生成侧

```text
L_aux_em = sum_k q_k * loc_loss_k
```

- 梯度路由:
  - loc head 在 aux 路径中冻结
  - `q_k` detach
  - 梯度只通过 `loc_loss_k` 回到 `x0_hat_k -> gen_head`

### Implemented experiment

| 配置 | 基础配置 | 主要改动 | 目的 |
|---|---|---|---|
| `exp21_dust2` | `exp17_2_dust2` | `aux_loc` 从单个 `x0_hat` 改为 `K` 个 candidate 的 detached responsibility-weighted expectation | 验证 multi-sample latent expectation 是否比单步 `aux_loc_loss` 更稳定 |

建议第一版最小设置：

```yaml
is_loc_aux_loss: True
is_aux_loc_em_loss: True
aux_loc_em_num_samples: 2
aux_loc_em_weight_mode: softmax_loss
aux_loc_em_candidate_tau: 1.0
aux_loc_em_backward_mode: weighted_all
aux_loc_em_share_loc_noise: True
```

### Current status

- `exp21_dust2` 已实现训练配置和 test 配置。
- 配置文件：
  - `csgo_configs/exp21_dust2.yaml`
  - `csgo_configs/test/exp21_dust2_gen.yaml`
  - `csgo_configs/test/exp21_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp21_dust2_loc.yaml`
- 第一版只实现 `weighted_all`，不包含 hard top1。
- 该方案应优先在 `exp17_2_dust2` 上验证，再决定是否迁移到 `exp20_dust2` 或 loc-aware REPA 主线。

### Key distinction from exp20_dust2

- `exp20_dust2`:
  - 主要修正 loc 主分支的输入分布
  - 让 loc head 见到更接近 `x0_hat` 的 noisy FPS
- `exp21_dust2`:
  - 主要修正 aux 分支的 latent expectation 近似
  - 用多个 `x0_hat_k` candidate 估计哪个 clean-view surrogate 更可信

## Uncertainty-weighted Residual Consistency Aux-Loc Line

### Design summary

`exp22_dust2` 是一个已实现实验，目标是把 aux-loc 的多次采样不确定性作为样本级权重，而不是区分“大家一起对”和“大家一起错”。核心假设是：

- `loc_loss` 本身负责判断对错并提供纠错梯度
- uncertainty weight 只负责识别多次 `x0_hat` 采样给出的 loc 训练信号是否一致
- 如果多次采样的 velocity residual 一致，则保留较高 aux 权重
- 如果多次采样的 velocity residual 分歧大，则降低该样本的 aux 权重

第一版建议只使用两个 gen-side samples：

```text
x0_hat_1, x0_hat_2
```

并强制共享 loc-side 随机量：

```text
same loc_time
same loc_noise
same x_t
same velocity target u_t
```

否则 uncertainty 会混入定位分支自身的 flow-matching 随机性。

### Weight definition

对同一样本的两个 `x0_hat` candidate，冻结 loc evaluator 后得到：

```text
v_1 = F_loc(x0_hat_1, map, x_t, loc_time)
v_2 = F_loc(x0_hat_2, map, x_t, loc_time)
u   = velocity target
```

定义 residual：

```text
r_1 = v_1 - u
r_2 = v_2 - u
```

使用 residual disagreement 作为 uncertainty：

```text
d = mean(abs(r_1 - r_2))
scale = 0.5 * (mean(abs(r_1)) + mean(abs(r_2))) + eps
d_norm = d / scale
w_unc = exp(- stopgrad(d_norm) / tau_unc)
```

最终 aux loss：

```text
L_aux_unc = stopgrad(w_unc) * 0.5 * (MSE(v_1, u) + MSE(v_2, u))
```

这个权重不直接判断 loss 大小：

- 两次都错但 residual 一致：
  - `d_norm` 小，`w_unc` 高
  - loss 大，模型仍会收到强纠错梯度
- 两次都对：
  - `d_norm` 小，`w_unc` 高
  - loss 小，不会产生过强更新
- 有对有错或方向不一致：
  - `d_norm` 大，`w_unc` 低
  - 降低不稳定 aux 信号的影响

### Implemented experiment

| 配置 | 基础配置 | 主要改动 | 目的 |
|---|---|---|---|
| `exp22_dust2` | `exp17_2_dust2` | 用两个 `x0_hat` candidate 的 loc velocity residual disagreement 计算样本级 uncertainty weight | 验证只抑制“不一致采样”是否比 EM-style candidate responsibility 更稳 |

建议第一版最小设置：

```yaml
is_loc_aux_loss: True
is_aux_loc_uncertainty_loss: True
aux_loc_unc_num_samples: 2
aux_loc_unc_metric: residual_l1_normed
aux_loc_unc_tau: 1.0
aux_loc_unc_min_weight: 0.05
aux_loc_unc_share_loc_noise: True
aux_loc_unc_eps: 1.0e-6
```

### Current status

- `exp22_dust2` 已实现训练配置和 test 配置。
- 配置文件：
  - `csgo_configs/exp22_dust2.yaml`
  - `csgo_configs/test/exp22_dust2_gen.yaml`
  - `csgo_configs/test/exp22_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp22_dust2_loc.yaml`
- 第一版只实现 `aux_loc_unc_num_samples=2` 和 `aux_loc_unc_metric=residual_l1_normed`。
- 第一版要求 `aux_loc_unc_share_loc_noise=True`，避免 uncertainty 混入 loc-side flow-matching 随机性。
- 该方案应优先独立对比 `exp17_2_dust2` 和 `exp21_dust2`，不要一开始叠加 `exp20_dust2` 的 noisy-loc matching。

### Comparison with exp21_dust2

| 维度 | `exp21_dust2` | `exp22_dust2` |
|---|---|---|
| 核心思想 | multi-sample latent expectation / EM-inspired responsibility | sample-level uncertainty gating |
| 权重对象 | candidate-level `q_k` | sample-level `w_unc` |
| 权重依据 | `loc_loss_k` 越小，candidate 越可信 | residual disagreement 越小，样本越稳定 |
| 是否区分一起对/一起错 | 会偏向 loss 小的 candidate | 不直接区分，对错由 loss 自身负责 |
| 主要抑制对象 | 低质量 candidate | 多次采样训练信号不一致的样本 |
| 推荐关系 | 更接近 EM-style posterior responsibility | 更接近 uncertainty-weighted consistency training |

## Combined EM-responsibility and Uncertainty-gated Aux-Loc Line

### Design summary

`exp23_dust2` 是一个已实现实验，目标是把 `exp21_dust2` 的 candidate-level responsibility 和 `exp22_dust2` 的 sample-level uncertainty gating 组合起来：

- `q_k` 负责回答：
  - 在同一样本的多个 `x0_hat_k` candidate 里，哪个 candidate 更可信
- `w_unc` 负责回答：
  - 这个样本的多次采样 loc 训练信号是否稳定，是否应该强更新

组合后的 aux objective：

```text
L_aux_combined = stopgrad(w_unc) * sum_k stopgrad(q_k) * loc_loss_k
```

其中：

```text
q_k = softmax(- stopgrad(loc_loss_k) / tau_candidate)
```

对于第一版 `K=2` 的情况，使用两个 candidate 的 velocity residual disagreement 计算样本级 uncertainty：

```text
r_1 = v_1 - u
r_2 = v_2 - u
d = mean(abs(r_1 - r_2))
scale = 0.5 * (mean(abs(r_1)) + mean(abs(r_2))) + eps
d_norm = d / scale
w_unc = exp(- stopgrad(d_norm) / tau_unc)
```

这里仍然要求两个 candidate 共享 loc-side 随机量：

```text
same loc_time
same loc_noise
same x_t
same velocity target u_t
```

否则 `w_unc` 会混入 loc 分支自身采样造成的不确定性。

### Interpretation

`exp23_dust2` 可以理解为两级权重：

| 权重 | 层级 | 作用 |
|---|---|---|
| `q_k` | candidate-level | 偏向 loc loss 更小、更像 clean-view surrogate 的 `x0_hat_k` |
| `w_unc` | sample-level | 降低多次采样 residual 不一致样本的整体 aux 强度 |

因此三类情况的行为为：

- 两次都对：
  - `loc_loss_k` 小
  - residual disagreement 小
  - `w_unc` 高，但总 loss 小
- 两次都错但方向一致：
  - `loc_loss_k` 大
  - residual disagreement 小
  - `w_unc` 高，模型收到稳定纠错梯度
- 有对有错或方向不一致：
  - `q_k` 会偏向较好的 candidate
  - `w_unc` 会降低整体样本权重
  - 该样本不会对生成侧产生过强、不稳定的 aux 更新

### Implemented experiment

| 配置 | 基础配置 | 主要改动 | 目的 |
|---|---|---|---|
| `exp23_dust2` | `exp17_2_dust2` | 同时启用 detached `q_k` 和 detached `w_unc` | 验证 candidate selection 与 sample reliability gating 是否互补 |
| `exp26_dust2` | `exp17_2_2_dust2` | 加入 `exp23` combined aux-loc，改为 `img_size=224` 和 `use_short_instruction=True`，不使用 warm-start / REPA / aux_gen | 在原始 `UniLIP-1B` 起训条件下隔离验证 224 输入、短 prompt 与 combined aux-loc 的组合效果 |
| `exp26_1_dust2` | `exp26_dust2` | `aux_loc_timestep_weight_type=low_noise_only`; `aux_loc_low_noise_sigma_max=0.45`; `aux_loc_timestep_weight_renorm=active_mean` | 验证 pose-level aux-loc 是否只应在生成侧低噪声、接近 clean image 的 timestep 上生效 |
| `exp26_2_dust2` | `exp26_dust2` | `aux_loc_timestep_weight_type=exp_sigma`; `aux_loc_exp_weight_lambda=5.0`; `aux_loc_timestep_weight_renorm=none` | 验证保留所有 timestep 但使用 `exp(-5*sigma)` 强化低噪声偏置是否比 hard gate 更稳定 |
| `exp27_dust2` | `exp26_dust2` | 关闭 `aux_loc_loss` / combined aux-loc，加入 current-loc-head perceptual loss，`smooth_l1` 对齐 action_dit_projector patch features；step < 2000 时 alpha=0，step >= 2000 时 alpha=0.1 | 验证不依赖外部 teacher 和 pose-level aux 的 feature-level 对齐是否能单独提供生成侧定位感知监督 |
| `exp27_1_dust2` | `exp27_dust2` | 在 patch-level `smooth_l1` loc perception loss 上加入 action_dit attention 权重；`teacher_gt + last layer + mean heads + mean_one + detach + loc_sampled`；实现使用轻量 wrapper，不用 hook | 验证 action token 对 und patch tokens 的注意力能否提供更聚焦且量级可比的生成侧定位感知监督 |
| `exp27_2_dust2` | `exp27_dust2` | 将 loc perception feature source 从 `action_dit_projector` 改为 `vision_tower`，直接对齐 `pred_pixels_input` 与 `gen_image` 的 FPS vision patch features；`smooth_l1`；无 attention weighting | 验证更浅的视觉 encoder patch feature 对齐是否足以提供生成侧定位感知监督 |
| `exp27_3_dust2` | `exp27_2_dust2` | 在 vision_tower patch-level `smooth_l1` loc perception loss 上加入 teacher_gt action_dit attention 权重；`teacher_gt + last layer + mean heads + mean_one + detach + loc_sampled`；复用轻量 wrapper，不用 hook | 验证 loc/action_dit 注意力能否让浅层 vision_tower feature 对齐更聚焦 |
| `exp28_dust2` | `exp26_1_dust2` | 保留 low-noise-only combined aux-loc，并加入 `exp27_3_dust2` 的 attention-weighted vision_tower loc perception；`alpha_loc_aux` 从 step 0 到 6000 线性 warmup 到 2.0，`alpha_loc_perception` 从 step 2000 开始为 0.1 | 验证 pose-level low-noise aux-loc 与 attention-focused patch feature 对齐是否互补 |

建议第一版最小设置：

```yaml
is_loc_aux_loss: True
is_aux_loc_combined_em_unc_loss: True
aux_loc_combined_num_samples: 2
aux_loc_combined_candidate_tau: 1.0
aux_loc_combined_unc_metric: residual_l1_normed
aux_loc_combined_unc_tau: 1.0
aux_loc_combined_unc_min_weight: 0.05
aux_loc_combined_share_loc_noise: True
aux_loc_combined_unc_eps: 1.0e-6
```

### Current status

- `exp23_dust2` 已实现训练配置和 test 配置。
- 配置文件：
  - `csgo_configs/exp23_dust2.yaml`
  - `csgo_configs/test/exp23_dust2_gen.yaml`
  - `csgo_configs/test/exp23_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp23_dust2_loc.yaml`
  - `csgo_configs/exp26_dust2.yaml`
  - `csgo_configs/test/exp26_dust2_gen.yaml`
  - `csgo_configs/test/exp26_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp26_dust2_loc.yaml`
  - `csgo_configs/exp26_1_dust2.yaml`
  - `csgo_configs/test/exp26_1_dust2_gen.yaml`
  - `csgo_configs/test/exp26_1_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp26_1_dust2_loc.yaml`
  - `csgo_configs/exp26_2_dust2.yaml`
  - `csgo_configs/test/exp26_2_dust2_gen.yaml`
  - `csgo_configs/test/exp26_2_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp26_2_dust2_loc.yaml`
  - `csgo_configs/exp27_dust2.yaml`
  - `csgo_configs/test/exp27_dust2_gen.yaml`
  - `csgo_configs/test/exp27_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp27_dust2_loc.yaml`
  - `csgo_configs/exp27_1_dust2.yaml`
  - `csgo_configs/test/exp27_1_dust2_gen.yaml`
  - `csgo_configs/test/exp27_1_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp27_1_dust2_loc.yaml`
  - `csgo_configs/exp27_2_dust2.yaml`
  - `csgo_configs/test/exp27_2_dust2_gen.yaml`
  - `csgo_configs/test/exp27_2_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp27_2_dust2_loc.yaml`
  - `csgo_configs/exp27_3_dust2.yaml`
  - `csgo_configs/test/exp27_3_dust2_gen.yaml`
  - `csgo_configs/test/exp27_3_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp27_3_dust2_loc.yaml`
  - `csgo_configs/exp28_dust2.yaml`
  - `csgo_configs/test/exp28_dust2_gen.yaml`
  - `csgo_configs/test/exp28_dust2_gen_conti.yaml`
  - `csgo_configs/test/exp28_dust2_loc.yaml`
- 第一版只实现 `aux_loc_combined_num_samples=2` 和 `aux_loc_combined_unc_metric=residual_l1_normed`。
- 第一版要求 `aux_loc_combined_share_loc_noise=True`，避免 uncertainty 混入 loc-side flow-matching 随机性。
- 该方案应在 `exp21_dust2` 和 `exp22_dust2` 的单独实验后再跑，避免无法判断收益来自 candidate responsibility 还是 uncertainty gating。

### Comparison with exp21_dust2 and exp22_dust2

| 维度 | `exp21_dust2` | `exp22_dust2` | `exp23_dust2` |
|---|---|---|---|
| 权重形式 | `sum_k q_k * loss_k` | `w_unc * mean_k(loss_k)` | `w_unc * sum_k q_k * loss_k` |
| 权重层级 | candidate-level | sample-level | candidate-level + sample-level |
| 主要目的 | 选择更可信的 `x0_hat_k` | 抑制采样不一致样本 | 同时选择可信 candidate 并抑制不稳定样本 |
| 风险 | 可能过度偏向当前 loc loss 小的 candidate | 不区分 candidate 质量 | 计算最重，且需要同时调 `tau_candidate` 和 `tau_unc` |
| 推荐定位 | EM-inspired ablation | uncertainty-gated ablation | 两者都有效后再验证的组合实验 |

## Implemented Loc-aware REPA Line

### Design summary

当前已实现的 loc-aware REPA 不是标准 REPA，而是使用定位分支的中间特征做 feature alignment：

- student:
  - 当前 `loc_aux_loss` 路径中
  - `action_dit_projector` 之后的 FPS prefix tokens
- teacher:
  - 冻结的 `exp14_dust2_loc`
  - 取同一层位的 loc feature extractor
- loss:
  - `1 - cosine(student_feature, teacher_feature)`
- 梯度路由:
  - 只更新生成侧
  - 不更新 loc teacher

### Implemented configs

| 配置 | 目的 | 关键改动 |
|---|---|---|
| `exp17_5_dust2` | 在 `exp17_2_dust2` 上加入独立 `loc-aware REPA` | `is_loc_repa_loss=True` |
| `exp20_dust2` | `exp17_2_dust2` 上加入 noisy-loc distribution matching | `is_noisy_loc_loss=True`; `noisy_loc_ratio=0.3`; `noisy_loc_image_source=latent_space`; `noisy_loc_sigma_sampling=gen_matched`; `noisy_loc_weight_type=linear_1m_sigma` |
| `exp17_6_dust2` | 在 `exp17_5_dust2` 上再打开 shared LLM tail | `train_shared_llm_tail_only=True` |
| `exp17_6_1_dust2` | 在 `exp17_6_dust2` 上把 shared LLM tail 从 2 层扩展到 6 层 | `shared_llm_tail_num_layers=6` |
| `exp17_6_2_dust2` | 在 `exp17_6_dust2` 上启用 12-layer shared tail 的 lora_only 方案 | `shared_llm_tail_num_layers=12`; `shared_llm_tail_lora_enabled=True`; `shared_llm_tail_lora_mode=lora_only` |

### Current notes

- `loc_repa teacher` 当前已经支持独立 frozen bundle。
- 当前代码已支持：
  - `train_shared_llm_tail_only`
  - `train_mm_projector_only`
- `exp17_6_dust2` 的一个已修复问题是：
  - non-registered teacher 在 forward 前需要按需迁移到当前 CUDA device。

## Traditional REPA Line

### Goal

回到更接近标准 REPA / iREPA 的方案：

- teacher:
  - `DINOv2`
  - 或 frozen `UniLIP vision_tower + multi_modal_projector`
- student:
  - 生成分支 `Sana DiT` 的中间层 hidden states
  - 首选 layer `6`，备选 layer `8`
- projector:
  - 原始 REPA: `three-layer MLP + SiLU`
  - iREPA 变体: `3x3 Conv + spatial normalization`
- loss:
  - `1 - cosine(student_feature, teacher_feature)`
- 对齐方式:
  - 默认使用 `patch-wise` alignment

### Current implementation status

当前工作区已经落地了 traditional REPA 的第一版实现，对应 `exp17_7_dust2`：

- 已实现：
  - `teacher = DINOv2`
  - `teacher = UniLIP vision`
  - `student = Sana DiT` 中间层 hidden states
  - `student_layer = 6`
  - `alignment = patch-wise`
  - `projector = three-layer MLP + SiLU`
  - `projector = conv_spatialnorm`
  - `L_repa` 只更新 `DiT + repa projector`
  - `repa_detach_condition = True`，避免 `repa_loss` 额外更新 `llm_connector/projector`
- 尚未实现：
  - 更通用的 `align_type` / teacher / projector 扩展

### Recommended default design

主方案先固定为：

- `teacher = DINOv2`
- `student_layer = 6`
- `alignment = patch-wise`
- `projector = three-layer MLP + SiLU`
- `L_repa` 只更新 `DiT + repa projector`
- 保持与 `aux_loc_loss` 解耦

其中 iREPA 风格的 student 侧处理可参考：

```python
# Conv projection instead of MLP
proj_layer = nn.Conv2d(D_in, D_out, kernel_size=3, padding=1)

# Spatial normalization on encoder features [B, T, D]
x = x - gamma * x.mean(dim=1, keepdim=True)
x = x / (x.std(dim=1, keepdim=True) + 1e-6)
```

## REPA + Loc-aware REPA Experiment Matrix

下面的 10 个实验是推荐推进顺序，不是全因子枚举。这样可以在较少预算下，逐步回答“teacher 选谁、REPA 与 aux_loc 是否互补、原始 REPA projector 和 iREPA 变体谁更好、DiT layer 取哪里、shared 是否有帮助”等问题。

| 序号 | 建议实验名 | 基础配置 | 主要改动 | 实验目的 |
|---|---|---|---|---|
| 1 | `exp17_2_dust2` | `exp17_2_dust2` | 无 | `dust2` unified 主 baseline |
| 2 | `exp17_5_dust2` | `exp17_2_dust2` | 加 loc-aware REPA | 验证 loc-aware REPA 是否有效 |
| 3 | `exp17_6_dust2` | `exp17_5_dust2` | 加 shared LLM tail | 验证 loc-aware REPA 在 shared tail 下是否进一步受益 |
| 4 | `exp17_7_dust2` | `exp17_2_dust2` | 传统 REPA, `teacher=DINOv2`, `layer=6`, `align=patch_wise`, `projector=mlp3_silu`, `aux_loc=off` | 验证标准 REPA 主方案本身是否有效 |
| 5 | `exp17_8_dust2` | `exp17_7_dust2` | 打开 `aux_loc` | 验证标准 REPA 与 `aux_loc` 是否互补 |
| 6 | `exp17_9_dust2` | `exp17_2_dust2` | 传统 REPA, `teacher=UniLIP vision`, `layer=8`, `align=patch_wise`, `projector=mlp3_silu`, `aux_loc=off` | 验证 domain-specific teacher 是否优于 DINOv2 |
| 7 | `exp17_10_dust2` | `exp17_9_dust2` | 打开 `aux_loc` | 验证 UniLIP vision teacher 与 `aux_loc` 联合时的效果 |
| 8 | `exp17_11_dust2` | `exp17_2_dust2` | iREPA, `teacher=DINOv2`, `layer=8`, `align=patch_wise`, `projector=conv_spatialnorm`, `aux_loc=off` | 验证 iREPA 是否可直接作为主方法替代经典 REPA |
| 9 | `exp17_12_dust2` | `exp17_11_dust2` | 打开 `aux_loc` | 在 iREPA 主方法基础上继续做组合或层位扩展 |
| 10 | `exp17_13_dust2` | `exp17_11_dust2` 或 `exp17_12_dust2` | 打开 shared LLM tail | 验证 shared tail 是否也能增强 iREPA 主方法 |

## REPA Config Diff Table

下面的差分表只描述每个实验相对其直接父实验需要改哪些配置键，便于后续逐个新建 yaml。

### A. Loc-aware REPA line

| 配置 | 相对父实验 | 需要改的键 | 目的 |
|---|---|---|---|
| `exp17_5_dust2` | `exp17_2_dust2` | `is_loc_repa_loss=True`; `alpha_loc_repa_loss=0.1`; `loc_repa_teacher_ckpt_path=...exp14_dust2_loc...`; `loc_repa_feature_type=action_prefix_tokens`; `loc_repa_loss_type=cosine`; `loc_repa_timestep_weight=linear_1m_sigma` | 加 loc-aware REPA |
| `exp17_6_dust2` | `exp17_5_dust2` | `train_shared_llm_tail_only=True`; `shared_llm_tail_num_layers=2`; `shared_llm_tail_lr=1.0e-5` | 在 loc-aware REPA 基础上增加 shared tail |
| `exp17_6_1_dust2` | `exp17_6_dust2` | `shared_llm_tail_num_layers=6` | 比较 6-layer shared tail full finetune 与 2-layer shared tail 的差异 |
| `exp17_6_2_dust2` | `exp17_6_dust2` | `shared_llm_tail_num_layers=12`; `shared_llm_tail_lora_enabled=True`; `shared_llm_tail_lora_mode=lora_only`; `shared_llm_tail_lora_r=16`; `shared_llm_tail_lora_alpha=32`; `shared_llm_tail_lora_dropout=0.05`; `shared_llm_tail_lora_lr=1.0e-4` | 资源受限下验证更深 shared tail 的 LoRA-only 替代路径 |

### B. Traditional REPA line

当前代码里已经落地并可直接在 yaml 中使用的 traditional REPA 键如下：

```yaml
is_repa_loss: True
alpha_repa_loss: 0.5
repa_teacher_type: dinov2
repa_teacher_name_or_path: facebook/dinov2-base
repa_teacher_input_size: 224
repa_teacher_hidden_size: 768
repa_dit_layer_idx: 6
repa_align_type: patch_wise
repa_expected_num_patches: 256
repa_projector_type: mlp3_silu
repa_mlp_num_layers: 3
repa_mlp_activation: silu
repa_mlp_hidden_ratio: 1.0
repa_detach_condition: True
```

当前版本代码约束：

- `repa_teacher_type` 目前支持 `dinov2` 和 `unilip_vision`
- `repa_align_type` 目前只支持 `patch_wise`
- `repa_projector_type` 目前支持 `mlp3_silu` 和 `conv_spatialnorm`
- `repa_mlp_activation` 目前只支持 `silu`
- `conv_spatialnorm` 目前只支持 `repa_conv_kernel_size=3`

如果是原始 REPA 的 projector，当前实现默认：

```yaml
repa_mlp_num_layers: 3
repa_mlp_activation: silu
repa_mlp_hidden_ratio: 1.0
```

如果使用 iREPA 变体的 projector，建议默认：

```yaml
repa_use_spatial_norm: True
repa_conv_kernel_size: 3
repa_spatial_norm_gamma: 1.0
```

| 配置 | 相对父实验 | 需要改的键 | 目的 |
|---|---|---|---|
| `exp17_7_dust2` | `exp17_2_dust2` | `is_repa_loss=True`; `alpha_repa_loss=0.5`; `repa_teacher_type=dinov2`; `repa_dit_layer_idx=6`; `repa_align_type=patch_wise`; `repa_projector_type=mlp3_silu`; `repa_mlp_num_layers=3`; `repa_mlp_activation=silu`; `repa_mlp_hidden_ratio=1.0`; `is_loc_aux_loss=False` | 标准 REPA 主起点 |
| `exp17_8_dust2` | `exp17_7_dust2` | `is_loc_aux_loss=True` | 观察 REPA 与 `aux_loc` 的互补性 |
| `exp17_9_dust2` | `exp17_2_dust2` | 同 `exp17_7_dust2`，但 `repa_teacher_type=unilip_vision` 且 `is_loc_aux_loss=False` | 用 UniLIP vision 作为 teacher 的 teacher ablation |
| `exp17_10_dust2` | `exp17_9_dust2` | `is_loc_aux_loss=True` | 观察 UniLIP vision teacher 与 `aux_loc` 的互补性 |
| `exp17_11_dust2` | `exp17_2_dust2` | `is_repa_loss=True`; `alpha_repa_loss=0.5`; `repa_teacher_type=dinov2`; `repa_teacher_name_or_path=facebook/dinov2-base`; `repa_teacher_input_size=224`; `repa_teacher_hidden_size=768`; `repa_dit_layer_idx=6`; `repa_align_type=patch_wise`; `repa_projector_type=conv_spatialnorm`; `repa_use_spatial_norm=True`; `repa_conv_kernel_size=3`; `repa_spatial_norm_gamma=1.0`; `is_loc_aux_loss=False` | iREPA 主方法起点 |
| `exp17_12_dust2` | `exp17_11_dust2` | `is_loc_aux_loss=True` 或 `repa_dit_layer_idx=8` | iREPA 的 `aux_loc` 组合或 layer 扩展 |
| `exp17_13_dust2` | `exp17_11_dust2` 或 `exp17_12_dust2` | `train_shared_llm_tail_only=True`; `shared_llm_tail_num_layers=2`; `shared_llm_tail_lr=1.0e-5` | shared tail 与 iREPA 主方法的交互实验 |

## Recommended Execution Order

推荐按下面顺序推进，不要一开始做全因子展开：

1. `exp17_2_dust2`
2. `exp17_4_dust2`
3. `exp19_dust2`
4. `exp20_dust2`
5. `exp21_dust2`
6. `exp22_dust2`
7. `exp23_dust2`
8. `exp26_dust2`
9. `exp26_1_dust2`
10. `exp26_2_dust2`
11. `exp27_dust2`
12. `exp27_1_dust2`
13. `exp27_2_dust2`
14. `exp27_3_dust2`
15. `exp28_dust2`
16. `exp17_5_dust2`
17. `exp17_6_dust2`
18. `exp17_7_dust2`
19. `exp17_8_dust2`
20. `exp17_9_dust2`
21. `exp17_10_dust2`
22. `exp17_11_dust2`
23. `exp17_12_dust2`
24. `exp17_13_dust2`

理由：

- 先固定当前 unified + loc-aware REPA 的锚点。
- 在引入 REPA 之前，先验证 noisy-loc 分布匹配是否能单独带来收益。
- 再验证 multi-sample EM-inspired aux-loc 是否能改善单个 `x0_hat` 的高方差问题。
- 再验证 uncertainty-weighted residual consistency 是否能用更简单的样本级 gating 抑制不稳定 aux 信号。
- 如果 `exp21_dust2` 和 `exp22_dust2` 任一有效，再验证二者组合是否互补。
- `exp26_dust2` 用原始 `UniLIP-1B` 起点补上 224 + short prompt + combined aux-loc 的干净对照，避免把 `exp24_dust2` 的收益误归因于 warm-start 或 REPA。
- `exp26_1_dust2` 在 `exp26_dust2` 基础上只保留低噪声生成 timestep 的 aux-loc，用来判断旧的高噪声 `1-sigma` 连续加权是否仍会引入无效或误导性定位梯度。
- `exp26_2_dust2` 在 `exp26_dust2` 基础上使用 `exp(-5*sigma)` 连续权重，保留高噪声 candidate 的弱监督，用来和 `exp26_1_dust2` 的 hard gate 区分。
- `exp27_dust2` 在 `exp26_dust2` 基础上关闭 pose-level aux-loc，只保留 current-loc-head perceptual alignment，用来判断 feature-level 监督本身是否成立。
- `exp27_1_dust2` 在 `exp27_dust2` 基础上加入 action_dit attention weighting，固定 `teacher_gt + last layer + mean heads + mean_one + detach + loc_sampled`；实现选型为轻量 wrapper，便于后续扩展 action token 构造、attention source 和 layer/head 聚合方式，不使用 hook。
- `exp27_2_dust2` 在 `exp27_dust2` 基础上把 perception feature source 前移到 `vision_tower`，无 attention weighting，用来隔离浅层视觉 patch 对齐本身的效果。
- `exp27_3_dust2` 在 `exp27_2_dust2` 基础上加入 teacher_gt action_dit attention weighting，检验浅层 vision_tower 对齐是否也能从 loc/action attention 聚焦中获益。
- `exp28_dust2` 是 `exp26_1_dust2` 和 `exp27_3_dust2` 的组合实验，应在两个分量实验之后再跑，用来判断 low-noise pose-level aux-loc 和 attention-weighted feature-level alignment 是否互补。
- 再验证传统 REPA 本身是否有效。
- 再比较 teacher。
- 然后直接验证 iREPA 作为主方法是否优于经典 REPA。
- 最后才引入 `aux_loc` 组合、layer 扩展和 shared tail，避免过早增加训练耦合。

## Practical Notes

- `exp17_3` 说明 shared `multi_modal_projector` 很危险，不应作为优先 shared 方案。
- `train_shared_llm_tail_only` 目前是更安全的 shared 位置。
- 传统 REPA 首版建议先只更新 `DiT + repa projector`，不要一开始同时更新 `llm_connector/projector`。
- 传统 REPA 这条线按论文默认设置，直接采用 `patch-wise` 对齐，不再单独保留 `mean(patch)` 主实验。
- `DINOv2` 应视作传统 REPA 主方案，`UniLIP vision` 应视作 domain-specific teacher ablation。

## Useful Paths

- `csgo_configs/exp14_dust2_gen.yaml`
- `csgo_configs/exp14_dust2_loc.yaml`
- `csgo_configs/exp17_2_dust2.yaml`
- `csgo_configs/exp19_dust2.yaml`
- `csgo_configs/exp20_dust2.yaml`
- `csgo_configs/exp21_dust2.yaml`
- `csgo_configs/exp22_dust2.yaml`
- `csgo_configs/exp23_dust2.yaml`
- `csgo_configs/exp24_dust2.yaml`
- `csgo_configs/exp26_dust2.yaml`
- `csgo_configs/exp26_1_dust2.yaml`
- `csgo_configs/exp26_2_dust2.yaml`
- `csgo_configs/exp27_dust2.yaml`
- `csgo_configs/exp27_1_dust2.yaml`
- `csgo_configs/exp27_2_dust2.yaml`
- `csgo_configs/exp27_3_dust2.yaml`
- `csgo_configs/exp28_dust2.yaml`
- `csgo_configs/exp17_4_dust2.yaml`
- `csgo_configs/exp17_4_1_dust2.yaml`
- `csgo_configs/exp17_5_dust2.yaml`
- `csgo_configs/exp17_6_dust2.yaml`
- `csgo_configs/exp17_6_1_dust2.yaml`
- `csgo_configs/exp17_6_2_dust2.yaml`
- `csgo_configs/exp17_7_dust2.yaml`
- `csgo_configs/exp18_dust2.yaml`
- `csgo_configs/exp18_1_dust2.yaml`
- `record.md`
- `train_csgo.py`
- `unilip/model/language_model/unified_unilip.py`
