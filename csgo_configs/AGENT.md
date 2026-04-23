# CSGO Experiment Guide

本文件用于整理 `csgo_configs` 下当前有效的实验脉络、每个实验的目的，以及后续 REPA / loc-aware REPA 实验的规划矩阵，便于后续在新线程中继续补配置、改代码和安排 ablation。

## Scope

当前重点讨论的是 `dust2` 上的 unified 训练路线，核心目标是回答三类问题：

1. 定位辅助生成是否有效。
2. 这种提升是否来自 `aux_loc_loss`，还是仅仅来自 joint multitask。
3. 在 unified 框架下，进一步引入表征对齐类监督时，哪种 teacher / 哪种表征挂点更有效。

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
| 6 | `exp17_9_dust2` | `exp17_2_dust2` | 传统 REPA, `teacher=UniLIP vision`, `layer=6`, `align=patch_wise`, `projector=mlp3_silu`, `aux_loc=off` | 验证 domain-specific teacher 是否优于 DINOv2 |
| 7 | `exp17_10_dust2` | `exp17_9_dust2` | 打开 `aux_loc` | 验证 UniLIP vision teacher 与 `aux_loc` 联合时的效果 |
| 8 | `exp17_11_dust2` | `exp17_2_dust2` | iREPA, `teacher=DINOv2`, `layer=6`, `align=patch_wise`, `projector=conv_spatialnorm`, `aux_loc=off` | 验证 iREPA 是否可直接作为主方法替代经典 REPA |
| 9 | `exp17_12_dust2` | `exp17_11_dust2` | 打开 `aux_loc` 或改 `layer=8` | 在 iREPA 主方法基础上继续做组合或层位扩展 |
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
5. `exp17_5_dust2`
6. `exp17_6_dust2`
7. `exp17_7_dust2`
8. `exp17_8_dust2`
9. `exp17_9_dust2`
10. `exp17_10_dust2`
11. `exp17_11_dust2`
12. `exp17_12_dust2`
13. `exp17_13_dust2`

理由：

- 先固定当前 unified + loc-aware REPA 的锚点。
- 在引入 REPA 之前，先验证 noisy-loc 分布匹配是否能单独带来收益。
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
