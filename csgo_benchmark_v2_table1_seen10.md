# csgo benchmark v2 Table 1（Seen-10）实验设计与表格维护规则

## 实验设计定位

Table 1 汇报 CSGO Benchmark v2 Seen-10 split 的定位、离散条件生成与连续条件生成结果。下表按 LaTeX 中有效数据行末尾注释记录实验标识；没有标识的模型保留 `--`。设计定位仅说明表中任务角色及原表类别。

| 类别 | Method | Experiment | 设计定位 |
|---|---|---|---|
| Vision-Language-Action (VLA) model | X-VLA | `X-VLA-fair-frozen-vl` | 外部 VLA 定位比较模型 |
| Vision-Language-Action (VLA) model | RDT | `--` | 外部 VLA 定位比较模型 |
| Vision-Language-Action (VLA) model | OpenVLA-OFT | `--` | 外部 VLA 定位比较模型 |
| VLA model with shared action prior | $\pi_{0.5}^{\dagger}$ | `pi0.5_exp32_loc_main_frozen_vl` | 使用共享动作先验的 VLA 定位比较模型 |
| Conditional generation model | OmniGen | `--` | 外部条件生成比较模型 |
| Conditional generation model | ControlAR | `csgo_seen10_exp32gen_aligned_peft` | 外部条件生成比较模型 |
| Text-image-to-image model | Lumina-Accessory | `--` | 外部文本图像到图像比较模型 |
| Unified multimodal model | Show-o2 | `--` | 外部统一多模态比较模型 |
| Unified multimodal model | Puffin | `--` | 外部统一多模态比较模型 |
| Ours | pose prediction baseline | `exp32_loc` | 定位单任务基线 |
| Ours | multi-task baseline | `exp32` | 定位、离散生成与连续生成联合任务基线 |
| Ours | aux. objs. baseline | `exp32_1` | 定位、离散生成与连续生成联合任务基线 |
| Ours | FPV synthesis baseline | `exp32_gen` | 离散生成与连续生成单任务基线 |

## 统计口径与维护规则

- 结果统计采用等地图宏平均（equal-map macro averaging）。除另有说明，所有外部模型均使用相同 Seen-10 数据进行训练或适配；这是实验协议说明，不表示训练已完成。
- 参数量于 **2026-10-19** 按本机实际代码、训练审计、checkpoint 张量形状及推理清单核对；下方 `exp31*`／`exp37*` 实测候选区于 **2026-11-09** 补充审计。三列统一用 **M = 10⁶ 个参数**，保留三位小数；精确整数和来源见文末“参数量核对依据”。
- **Total params** 为该任务推理实例实际加载的唯一参数，包括冻结模块；**Active params** 为完成该任务前向时实际执行的参数张量并集。按整个 embedding／参数张量计数，不按一次输入命中的 token 行计数；CFG、去噪步数、连续帧及重复图像编码均不重复累加。已执行但输出未使用的模块仍计入 Active，根本未调用的模块不计入。
- **Trainable params** 为该 Seen-10 配方中优化器覆盖且可由训练目标更新的参数并集；联合训练在各任务行重复同一训练总数。静态确定与所有训练损失断开的参数不算“更新参数”，并在文末列出与声明预算的差额。`§` 配置组报告可训练预算，不表示已经完成适配。
- 去重 tied/shared 参数；不计 buffer、优化器状态、EMA 副本、缓存特征及构造后已释放的临时模型。LoRA 若在实际推理前 merge 并卸载，Total／Active 按合并后结构计；Trainable 仍按训练时适配参数计。离散与连续生成沿用同一推理结构，三列一致。
- **§**：已有同配方结构／参数审计，但尚未确认与表内正式评测实例对应；Lumina-Accessory 本机仅有结构验收记录，Puffin 为 aligned 配置审计且本机缺少匹配表中指标的正式产物。**‡**：Show-o2 采用本机 `csgo_seen10_exp32gen_aligned_v2_final` 的正式运行参数，表中 Experiment 仍为 `--`，作为具名候选配置列示，待绑定后再去掉标记。
- `T-Warp` 和 `T-Diff` 分别指 Temporal_Warping_Error 和 Temporal_Difference_Error。$\pi_{0.5}^{\dagger}$ 表示模型使用共享的、与 $\pi_{0.5}$ 兼容的动作先验。
- `--` 表示未提供。仅将未注释的有效数据行纳入结果；注释掉的旧结果及整行注释模型均不纳入。
- LaTeX 生成表的分组标题仍有旧 SSIM 列结构残留，以下按实际有效指标标题和数值整理，不补入 SSIM；OmniGen 的多余空占位也不作为指标。
- 三张任务结果表保留原 LaTeX 模型顺序；后续更新原位填充，不按成绩重新排序。

# 实验进度

本表只根据 Table 1 中的有效数值记录 Metric 报告情况，不用其他文档中的运行信息推断训练或推理状态。`未提供` 不等同于 `未开始`；重复出现的联合任务实验合并为一行。

| 实验版本 | 训练 | 推理 | Metric 计算 |
|---|---|---|---|
| `X-VLA-fair-frozen-vl` | 未提供 | 未提供 | ✅ Seen-10 定位指标已报告 |
| `--`（RDT） | 未提供 | 未提供 | 未提供（表中为 `--`） |
| `--`（OpenVLA-OFT） | 未提供 | 未提供 | 未提供（表中为 `--`） |
| `pi0.5_exp32_loc_main_frozen_vl` | 未提供 | 未提供 | ✅ Seen-10 定位指标已报告 |
| `exp32_loc` | 未提供 | 未提供 | ✅ Seen-10 定位指标已报告 |
| `exp32` | 未提供 | 未提供 | ✅ Seen-10 定位/离散生成/连续生成指标已报告 |
| `exp32_1` | 未提供 | 未提供 | ✅ Seen-10 定位/离散生成/连续生成指标已报告 |
| `--`（OmniGen） | 未提供 | 未提供 | 未提供（表中为 `--`） |
| `csgo_seen10_exp32gen_aligned_peft` | 未提供 | 未提供 | ✅ Seen-10 离散生成/连续生成指标已报告 |
| `--`（Lumina-Accessory） | 未提供 | 未提供 | 未提供（表中为 `--`） |
| `--`（Show-o2） | 未提供 | 未提供 | 未提供（表中为 `--`） |
| `--`（Puffin） | 未提供 | 未提供 | 未提供（表中为 `--`） |
| `exp32_gen` | 未提供 | 未提供 | ✅ Seen-10 离散生成/连续生成指标已报告 |

# 实验结果表

## 定位

| Setting | Task | Method | Experiment | Total params | Active params | Trainable params | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Localization | X-VLA | `X-VLA-fair-frozen-vl` | 879.482 M | 879.482 M | 25.603 M | 65.969 | 4.162 | 2.908 | 27.748 |
| Seen-10 | Localization | RDT | `csgo_aligned_aug_v1` | 1681.640 M | 1681.640 M | 73.165 M | 43.251 | 2.783 | 2.306 | 20.779 |
| Seen-10 | Localization | OpenVLA-OFT | `csgo_seen10_aligned_v2` | 7739.887 M | 7724.644 M | 197.493 M | 36.956 | 2.264 | 1.146 | 17.137 |
| Seen-10 | Localization | $\pi_{0.5}^{\dagger}$ | `pi0.5_exp32_loc_main_frozen_vl` | 3431.291 M | 3431.291 M | 80.023 M | 53.542 | 3.453 | 1.251 | 25.450 |
| Seen-10 | Localization | pose prediction baseline | `exp32_loc` | 2269.423 M | 1285.918 M | 55.689 M | 48.961 | 3.006 | 2.970 | 26.047 |
| Seen-10 | Localization | multi-task baseline | `exp32` | 2288.441 M | 1285.918 M | 77.003 M | 123.194 | 6.38 | 3.352 | 55.449 |
| Seen-10 | Localization | aux. objs. baseline | `exp32_1` | 2288.441 M | 1285.918 M | 77.003 M | 63.0377 | 3.5604 | 3.0882 | 31.6713 |

## 离散生成

| Setting | Task | Method | Experiment | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | Boundary_F1↑ | FID↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Discrete generation | OmniGen | `exp32gen_aligned` | 7805.604 M | 7136.920 M | 5.107 M | 11.9736 | 0.6915 | 0.5057 | 83.4449 |
| Seen-10 | Discrete generation | ControlAR | `exp32gen_aligned_peft` | 910.198 M | 859.926 M | 69.317 M | 13.3055 | 0.6230 | 0.5103 | 30.2869 |
| Seen-10 | Discrete generation | Lumina-Accessory | `--` | 5710.672 M§ | 5710.670 M§ | 227.964 M§ | 12.823 | 0.643 | 0.511 | 31.098 |
| Seen-10 | Discrete generation | Show-o2 | `--` | 3263.077 M‡ | 3030.113 M‡ | 336.481 M‡ | 14.539 | 0.591 | 0.529 | 37.430 |
| Seen-10 | Discrete generation | Puffin | `exp32gen_aligned` | 4560.858 M§ | 3898.988 M§ | 88.824 M§ | 14.3186 | 0.5924 | 0.5357 | 33.2448 |
| Seen-10 | Discrete generation | FPV synthesis baseline | `exp32_gen` | 1822.416 M | 1681.653 M | 38.910 M | 13.965 | 0.6106 | 0.5163 | 35.584 |
| Seen-10 | Discrete generation | multi-task baseline | `exp32` | 1822.416 M | 1681.653 M | 77.003 M | 14.1528 | 0.6035 | 0.5229 | 36.6005 |
| Seen-10 | Discrete generation | aux. objs. baseline | `exp32_1` | 1822.416 M | 1681.653 M | 77.003 M | 13.9070 | 0.6145 | 0.5151 | 36.2159 |

## 连续生成

| Setting | Task | Method | Experiment | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | T-Warp↓ | T-Diff↓ | FVD↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Continuous generation | OmniGen | `exp32gen_aligned` | 7805.604 M | 7136.920 M | 5.107 M | 12.7187 | 0.6993 | 24.9154 | 33.0017 | 1631.3874 |
| Seen-10 | Continuous generation | ControlAR | `exp32gen_aligned_peft` | 910.198 M | 859.926 M | 69.317 M | 13.9832 | 0.6157 | 39.2998 | 45.1919 | 951.4971 |
| Seen-10 | Continuous generation | Lumina-Accessory | `--` | 5710.672 M§ | 5710.670 M§ | 227.964 M§ | 13.722 | 0.634 | 38.103 | 44.423 | 932.545 |
| Seen-10 | Continuous generation | Show-o2 | `--` | 3263.077 M‡ | 3030.113 M‡ | 336.481 M‡ | 15.446 | 0.577 | 32.709 | 39.043 | 887.303 |
| Seen-10 | Continuous generation | Puffin | `exp32gen_aligned` | 4560.858 M§ | 3898.988 M§ | 88.824 M§ | 15.4657 | 0.5679 | 28.1975 | 35.2320 | 844.2427 |
| Seen-10 | Continuous generation | FPV synthesis baseline | `exp32_gen` | 1822.416 M | 1681.653 M | 38.910 M | 14.342 | 0.5929 | 35.698 | 41.867 | 848.071 |
| Seen-10 | Continuous generation | multi-task baseline | `exp32` | 1822.416 M | 1681.653 M | 77.003 M | 14.5725 | 0.5826 | 34.0925 | 40.3596 | 848.4586 |
| Seen-10 | Continuous generation | aux. objs. baseline | `exp32_1` | 1822.416 M | 1681.653 M | 77.003 M | 14.2285 | 0.5981 | 36.1777 | 42.2848 | 891.6025 |

## 参数量核对依据（2026-10-19）

下表给出未舍入的整数。这里的核对只更新参数统计，不改变上面的指标、实验标识或训练／推理进度。采用已保存训练审计中的 `named_parameters`／优化器成员、实际推理加载代码和 checkpoint 形状交叉核验，没有启动训练、推理或评测。

| 方法／配置与任务 | Total（个） | Active（个） | Trainable（个） | 主要本机依据 |
|---|---:|---:|---:|---|
| X-VLA；Localization | 879,482,456 | 879,482,456 | 25,602,648 | [训练参数审计](../X-VLA/outputs/csgo_benchmark_v2_seen10/X-VLA-fair-frozen-vl/seed_0/parameter_audit.json)、[step_00019500 原生权重](../X-VLA/outputs/csgo_benchmark_v2_seen10/X-VLA-fair-frozen-vl/seed_0/checkpoints/step_00019500/model.safetensors)、[推理加载](../X-VLA/infer_seen10.py) |
| RDT；Localization | 1,681,639,616 | 1,681,639,616 | 73,164,928 | [训练元数据](../RoboticsDiffusionTransformer/outputs/csgo_aligned_aug_v1/RDT/seed_42/training_metadata.json)、[checkpoint-19500](../RoboticsDiffusionTransformer/outputs/csgo_aligned_aug_v1/RDT/seed_42/checkpoints/checkpoint-19500/model.safetensors)、[推理加载](../RoboticsDiffusionTransformer/infer_seen10.py) |
| OpenVLA-OFT；Localization | 7,739,887,045 | 7,724,644,341 | 197,492,741 | [正式参数审计](../openvla-oft/outputs/csgo_benchmark_v2_seen10_aligned_v2/OpenVLA-OFT/seed_42/parameter_audit.json)、[定位前向与 L1 loss](../openvla-oft/csgo_seen10/model.py)、[视觉中间层前向](../openvla-oft/prismatic/extern/hf/modeling_prismatic.py) |
| π₀.₅；Localization | 3,431,290,640 | 3,431,290,640 | 80,022,560 | [19500 参数树元数据](../openpi/outputs/csgo_benchmark_v2_seen10/pi0.5_exp32_loc_main_frozen_vl/seed_0/checkpoints/19500/params/_METADATA)、[实际推理清单](../openpi/outputs/csgo_benchmark_v2_seen10/pi0.5_exp32_loc_main_frozen_vl/seed_0/localization_late/inference_manifest.json)、[冻结策略](../openpi/src/openpi/csgo/model.py) |
| OmniGen 表行，实际 OmniGen2；两类生成 | 7,805,604,059 | 7,136,919,771 | 5,107,200 | [正式参数审计](../OmniGen2/outputs/csgo_seen10_exp32gen_aligned/OmniGen2/seed_42/train/parameter_audit.json)、[正式推理清单](../OmniGen2/outputs/csgo_seen10_exp32gen_aligned/OmniGen2/seed_42/predictions/late/discrete/inference_manifest.json)、[加载与 LoRA fusion](../OmniGen2/infer_seen10.py) |
| ControlAR PEFT；两类生成 | 910,198,283 | 859,926,147 | 69,316,608 | [正式训练参数审计](../ControlAR/outputs/csgo_seen10_exp32gen_aligned_peft/ControlAR/seed_42/audits/trainable_parameters.json)、[正式推理清单](../ControlAR/outputs/csgo_seen10_exp32gen_aligned_peft/ControlAR/seed_42/predictions/late/inference_seed_42/inference_manifest.json)、[merge 与 VQ 加载](../ControlAR/infer_seen10.py) |
| Lumina-Accessory§；`csgo_seen10_exp32gen_aligned` 配置；两类生成 | 5,710,671,907 | 5,710,669,603 | 227,963,968 | [真实权重 CPU 结构验收](../Lumina-Accessory/outputs/implementation_audit/real_model/trainable_parameter_audit.md)、[配置](../Lumina-Accessory/configs/csgo_seen10_exp32gen_aligned.json)、[组件加载](../Lumina-Accessory/csgo_seen10/model.py) |
| Show-o2‡；`csgo_seen10_exp32gen_aligned_v2_final`；两类生成 | 3,263,077,075 | 3,030,113,491 | 336,481,088 | [正式参数审计](../Show-o/outputs/csgo_benchmark_v2_aligned/csgo_seen10_exp32gen_aligned_v2_final/seed_42/parameter_audit.json)、[late checkpoint 绑定](../Show-o/outputs/csgo_benchmark_v2_aligned/csgo_seen10_exp32gen_aligned_v2_final/seed_42/predictions/late/checkpoint_binding.json)、[生成前向](../Show-o/show-o2/models/modeling_showo2_qwen2_5.py) |
| Puffin§；`csgo_seen10_exp32gen_aligned` 配置；两类生成 | 4,560,857,827 | 3,898,988,259 | 88,823,808 | [aligned 配置](../Puffin/Puffin/configs/pipelines/csgo_seen10_exp32gen_aligned.py)、[同配方 smoke 参数审计](../Puffin/Puffin/outputs/aligned_smoke_20260927_OgjlCp/continuous/seed_42/parameter_audit.json)、[aligned 推理加载](../Puffin/Puffin/infer_seen10.py) |
| UniLIP `exp32_loc`；Localization | 2,269,423,363 | 1,285,918,368 | 55,688,992 | [训练日志](logs/csgo_1b/exp32_loc/train_20260902_014912/train.log)、[实际权重](outputs/csgo_1b/exp32_loc/model.safetensors)、[推理清单](outputs_loc/benchmark_v2/exp32_loc/seen/inference_manifest.json) |
| UniLIP `exp32_gen`；两类生成 | 1,822,415,843 | 1,681,653,155 | 38,910,208 | [训练日志](logs/csgo_1b/exp32_gen/train_20260902_014907/train.log)、[实际权重](outputs/csgo_1b/exp32_gen/model.safetensors)、[推理清单](outputs_eval/benchmark_v2/exp32_gen/seen/discrete/inference_manifest.json) |
| UniLIP `exp32`／`exp32_1`；Localization | 2,288,441,091 | 1,285,918,368 | 77,002,784 | [exp32 权重](outputs/csgo_1b/exp32/model.safetensors)、[exp32_1 权重](outputs/csgo_1b/exp32_1/model.safetensors)、[exp32_1 训练参数逐项日志](logs/csgo_1b/exp32_1/train_20260920_183417/train.log)、[定位加载](eval_csgo_loc.py) |
| UniLIP `exp32`／`exp32_1`；两类生成 | 1,822,415,843 | 1,681,653,155 | 77,002,784 | 同上权重；[exp32 配置](csgo_configs/exp32.yaml)与 [exp32_1 配置](csgo_configs/exp32_1.yaml)的可训练结构一致；[生成加载](eval_csgo.py)、[生成／定位前向](unilip/model/language_model/unified_unilip.py) |

关键计算与差异：

- **X-VLA**：推理加载合并后的原生模型，不加载训练用 PEFT 包装；训练审计总量 905,085,104 不能直接填推理 Total。权重文件中 tied token embedding 去重一次，并排除 `final_logits_bias`（51,289）和 `visual_temporal_embed.pos_idx_to_embed`（204,800）两个 buffer。Florence decoder／lm_head 已从模型删除；保留模块均参与定位前向。
- **RDT**：RDT 本体 1,253,414,016 加实际加载的 SigLIP 视觉模块 428,225,600。推理读取 T5 语言特征缓存，不加载 T5 参数，因此本口径不计 T5，也不计缓存本身。SigLIP pooling head 虽然输出未被采用，但前向确实执行，仍计入 Active。
- **OpenVLA-OFT**：视觉前向采用 `get_intermediate_layers(norm=False)`，跳过两个最终 norm 和 SigLIP attention pool，Active 扣 15,242,704。语言输出头实际计算 logits，所以计入 Active；但定位训练只反传 action L1，语言输出头的 LoRA 1,157,120 与训练目标断开。故主表有效 Trainable = 审计声明预算 198,649,861 − 1,157,120 = **197,492,741**；其余实验按相同可更新参数口径统计。
- **OmniGen2**：推理 Total = Qwen2.5-VL 3,754,622,976 + 合并 LoRA 后 Transformer 3,967,161,400 + VAE 83,819,683。文本 prompt 路径不调用 Qwen 的视觉塔，Active 扣 668,684,288；雷达与 FPV 分别使用 VAE encoder／decoder，两者都计。表中“OmniGen”名称保留，实际实验身份在此明确为 **OmniGen2**。
- **ControlAR**：Total = 训练审计模型 866,921,344 − 合并卸载的 LoRA 28,606,464 + 独立 VQ 71,883,403。Active 再扣未调用的 condition embedding 20,971,520、DINO mask token 384、VQ encoder 29,298,176 和 quant_conv 2,056。上述未调用 embedding／mask token 在训练审计中已冻结且不属于 optimizer，不需再从 Trainable 扣除。
- **Lumina-Accessory§**：Total = 含未合并 LoRA 的 DiT 3,012,510,336 + Gemma AutoModel 2,614,341,888 + VAE 83,819,683；Active 仅扣未调用的 DiT `norm_final.weight` 2,304。已验收配置含 227,963,968 个可训练参数，但正式配置 `run_formal=false`，本次未找到正式训练产物；该值是配置预算。
- **Show-o2‡**：Total = 含未合并 LoRA 的 backbone 3,136,184,544 + WanVAE 126,892,531；生成直接调用底层模型，Active 扣未调用的 `lm_head` 232,963,584。该 final 配方已将输入 embedding 与 lm_head 克隆解耦，参数审计确认无存储别名，Total 应分别计数。现有正式 checkpoint／预测绑定支持该候选配方，不能据此自动改写本表尚为空的实验标识或生成指标。
- **Puffin§**：同配方审计 Total 4,560,857,827；文本 pose、`reasoning=False` 的生成路径使用 VAE 处理雷达，RADIO 视觉编码器 651,642,880 和视觉 projector 10,226,688 闲置，故 Active 为 3,898,988,259；LoRA 保持未合并。当前本机可找到的完整 legacy 评测与表中指标不符，未以 legacy 权重冒充 aligned 正式产物；此处列的是 aligned 配置参数，待对应 checkpoint／清单同步后复核。
- **UniLIP**：定位 Active = 视觉塔 304,012,288 + 多模态投影 4,482,816 + LLM（含 LoRA）511,398,016 + 定位私有模块 466,025,248 = 1,285,918,368。生成入口只构建生成实例，没有调用 `initialize_localization_modules`，所以相同联合 checkpoint 在生成任务的 **Total 小于定位任务**；这不是把完整训练模型声称为更小模型。生成 Total 1,822,415,843，Active 再扣从未调用的 `lm_head` 135,903,488 和 VAE `down_blocks/down_mlp` 4,859,200，仅保留实际 decoder。完整训练 checkpoint 参数量分别为：`exp32_loc` 2,269,423,363，`exp32_gen` 2,274,580,227，`exp32/exp32_1` 2,288,441,091。主表 Total 始终使用任务推理实例口径；辅助 loss 未引入额外推理模块，联合两组的训练参数并集相同。

## exp37 新实验身份与 Table 1 实测候选（2026-11-04）

当前为1次Stage1＋5次Stage2，共六组。2026-11-09本机核查：原四组 Stage2 的 batch8 最终 test 摘要共9份；`exp37_s2_both_b1` 新增最终 test 三项，定位推理 batch32、离散／连续生成推理 batch16，合计12份不同 batch 的 test 摘要。B1 的 local4000／6000 定位（batch32）与离散生成（batch16）validation 也已同步。B1 [训练状态](outputs/csgo_1b/exp37_s2_both_b1/trainer_state.json)记录 local6000/6000；最终 checkpoint 参数头部、训练日志和评测清单已核对。B2 的 validation 仍仅有 local4000 定位（batch1）；`exp37_s2_gen` 的 batch8 离散指标在CPU计算、连续指标在CUDA计算。完整结果、来源和跨 batch 限制见[主实验文档 B1 补充](csgo_benchmark_v2_experiments_results.md#exp37-b1-seen10-update-20261109)，原四组 batch8 队列历史见[专题](CSGO_BENCHMARK_V2_EXP37_SEEN10.md#exp37-bs8-update-20261109)。本实测区不覆盖上方历史Table 1行，不预先指定最终Ours；`exp31*`／`exp37*` 参数量见[本节参数审计](#exp31-exp37-parameter-audit-20261109)。

| 候选实验 | 任务 | 正式checkpoint | 当前状态 | 参数量 |
|---|---|---|---|---|
| `exp37` | 定位／离散／连续 | Stage1 final19550，仅参照 | ✅ 训练、validation、test完成 | [已核对](#exp31-exp37-parameter-audit-20261109) |
| `exp37_s2_gen` | 离散／连续 | local6000 | 训练、validation已完成；batch8离散／连续最终test✅（离散CPU评测、连续CUDA评测） | [已核对](#exp31-exp37-parameter-audit-20261109) |
| `exp37_s2_loc` | 定位 | local6000 | 训练、validation已完成；batch8最终test定位已评 | [已核对](#exp31-exp37-parameter-audit-20261109) |
| `exp37_s2` | 定位／离散／连续 | local6000 | 训练、validation已完成；batch8最终test三项已评 | [已核对](#exp31-exp37-parameter-audit-20261109) |
| `exp37_s2_both` | 定位／离散／连续，P4＋U2 | local6000 | 训练及test三项✅（batch8）；local4000定位validation已评（batch1），其余validation未同步 | [已核对](#exp31-exp37-parameter-audit-20261109) |
| `exp37_s2_both_b1` | 定位／离散／连续，P3＋U2 | local6000（已完成） | ✅ validation4000／6000定位＋离散；test6000三项 | [最终权重](outputs/csgo_1b/exp37_s2_both_b1/checkpoint-6000/model.safetensors)、[训练日志](logs/csgo_1b/exp37_s2_both_b1/train_20261103_152114/train.log) |

2026-11-09补充：下方三张主表合并外部定位／生成基线、完整 Seen-10 的 `exp31*`／`exp32*` 与 `exp37*` 最终test结果；batch8中尚无指标摘要的项目按实际状态留空标记。本次替换的原 batch1 结果单列于各主表之后。`exp31*` 为 full-head，`exp32*` 为 LoRA；表中 joint＋aux 均表示同时开启 perception 与 aux-loc。只纳入各实验适用的任务，不纳入三地图消融、CrossMap／Unseen、validation 或中途checkpoint；保持外部模型原有顺序，再列各 UniLIP 系列，不按指标排序。

已链接原始JSON的指标采用完整十图等地图宏平均：定位与离散test各20000条，连续test为200×64=12800帧，统一保留四位小数。Lumina-Accessory／Puffin 的正式产物尚未与原表绑定，暂沿用[上方历史表](#实验结果表)的报告值及原精度，checkpoint／batch不作推断。Show-o2 的 `csgo_seen10_exp32gen_aligned_v2_final` 候选JSON与原表指标吻合，但原表实验标识仍为 `--`，保留‡；外部模型与 `exp32*` 参数量沿用[既有审计](#参数量核对依据2026-10-19)及§／‡标记；`exp31*`、`exp37*` 参数量见[2026-11-09补充审计](#exp31-exp37-parameter-audit-20261109)。B1 与 B2 的 test 推理 batch 不同，生成清单还记录 active LoRA dropout，不能将差异单独归因于 perception attention 开关。

本次核对后的元数据与比较边界：

- `exp32`／`exp32_1` 的最终步数按根目录 [exp32训练状态](outputs/csgo_1b/exp32/trainer_state.json)／[exp32_1训练状态](outputs/csgo_1b/exp32_1/trainer_state.json)记为 **19550**；各任务推理清单均指向对应根目录最终模型。旧主表中的19500不沿用。`exp32_1` 离散LPIPS原始值为0.6145836，新表按四位舍入记 **0.6146**（旧表0.6145）。
- `exp31*`／`exp32*` 推理清单所指测试配置均为batch1，旧清单未独立保存实际batch，故列为“1（配置）”。OmniGen2列16为初始batch，支持OOM回退；π0.5和Show-o2未找到推理batch记录，训练batch不作为替代。`exp31*`／`exp32*`／`exp37*` 为推理seed42；外部结果沿用各自清单，其中π0.5为seed0，X-VLA训练seed0、推理seed42。
- `exp37*` 使用minimal资产。单任务Stage2分别接旧 `exp32_gen@19500`／`exp32_loc@19500`，joint Stage2接 `exp37@19550`，均续训6000步；两阶段总预算需单独披露，不能主张与旧单阶段基线等FLOPs。full-head与LoRA也不是仅改变辅助loss的消融。
- `exp37_s2_both`、`exp37_s2`、`exp37_s2_gen` 与 `exp37_s2_loc` 的适用最终test主表行均按推理batch8列示；`exp37` 仍为batch1参照，原batch1结果在各主表后历史留存。`exp37_s2_gen` 的batch8离散与连续summary均已齐，适用test两项完成；仍需结合全部候选决定最终Ours。离散三组使用相同metric_profile、batch1指标计算及预处理；gen-only离散指标在CPU计算，joint／both离散指标在GPU计算，因此设备间离散数值不宣称严格等价。连续三组均在CUDA评测、推理batch8；gen-only的九项核心指标均优于joint与both，FVD为822.6760／928.6481／922.6230（gen-only／joint／both）。当前报告的离散七项核心指标中，gen-only均优于joint与both；joint与both差距小且方向有取舍。定位上both相对joint的XY／Z／Yaw距离更低、Pitch距离更高；连续生成中gen-only九项核心指标均较优，joint与both接近，未显示联合或辅助全面优于单任务。生成存在随batch改变的活跃Dropout随机性；both的validation与最终test分析仍见本节开头链接。

### exp37 定位（最终test）

| Method／训练方式 | Experiment | Checkpoint | 推理 batch | Total params | Active params | Trainable params | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 来源 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| X-VLA | `X-VLA-fair-frozen-vl` | final 19500 | 16 | 879.482 M | 879.482 M | 25.603 M | 65.9693 | 4.1617 | 2.9083 | 27.7481 | [JSON](../X-VLA/outputs/csgo_benchmark_v2_seen10/X-VLA-fair-frozen-vl/seed_0/metrics/localization/summary_equal_map.json) |
| RDT | `csgo_aligned_aug_v1` | final 19500 | 1 | 1681.640 M | 1681.640 M | 73.165 M | 43.2506 | 2.7827 | 2.3061 | 20.7786 | [JSON](../RoboticsDiffusionTransformer/outputs/csgo_aligned_aug_v1/RDT/seed_42/evaluation/localization/summary_equal_map.json) |
| OpenVLA-OFT | `csgo_seen10_aligned_v2` | final 19500 | 1 | 7739.887 M | 7724.644 M | 197.493 M | 36.9560 | 2.2642 | 1.1460 | 17.1370 | [JSON](../openvla-oft/outputs/csgo_benchmark_v2_seen10_aligned_v2/OpenVLA-OFT/seed_42/evaluation/localization/summary_equal_map.json) |
| $\pi_{0.5}^{\dagger}$ | `pi0.5_exp32_loc_main_frozen_vl` | final 19500 | 未记录 | 3431.291 M | 3431.291 M | 80.023 M | 53.5425 | 3.4532 | 1.2515 | 25.4504 | [JSON](../openpi/outputs/csgo_benchmark_v2_seen10/pi0.5_exp32_loc_main_frozen_vl/seed_0/evaluation/localization_late/summary_equal_map.json) |
| UniLIP full-head · loc-only | `exp31_loc` | final 19500 | 1（配置） | 2237.966 M | 1254.461 M | 452.164 M | 79.2937 | 4.1339 | 2.7075 | 36.1728 | [JSON](outputs_loc/benchmark_v2/exp31_loc/seen/benchmark_csgo_v2_loc.json) |
| UniLIP full-head · joint | `exp31_1` | final 19500 | 1（配置） | 2237.966 M | 1254.461 M | 1135.684 M | 82.7500 | 4.3542 | 2.9693 | 37.1299 | [JSON](outputs_loc/benchmark_v2/exp31_1/seen/benchmark_csgo_v2_loc.json) |
| UniLIP full-head · joint＋aux | `exp31` | final 19550 | 1（配置） | 2237.966 M | 1254.461 M | 1135.684 M | 99.0970 | 4.9604 | 3.2192 | 45.1100 | [JSON](outputs_loc/benchmark_v2/exp31/seen/benchmark_csgo_v2_loc.json) |
| UniLIP LoRA · loc-only | `exp32_loc` | final 19500 | 1（配置） | 2269.423 M | 1285.918 M | 55.689 M | 48.9611 | 3.0059 | 2.9704 | 26.0468 | [JSON](outputs_loc/benchmark_v2/exp32_loc/seen/benchmark_csgo_v2_loc.json) |
| UniLIP LoRA · joint | `exp32` | final 19550 | 1（配置） | 2288.441 M | 1285.918 M | 77.003 M | 123.1938 | 6.3845 | 3.3523 | 55.4486 | [JSON](outputs_loc/benchmark_v2/exp32/seen/benchmark_csgo_v2_loc.json) |
| UniLIP LoRA · joint＋aux | `exp32_1` | final 19550 | 1（配置） | 2288.441 M | 1285.918 M | 77.003 M | 63.0377 | 3.5604 | 3.0882 | 31.6713 | [JSON](outputs_loc/benchmark_v2/exp32_1/seen/benchmark_csgo_v2_loc.json) |
| UniLIP · Stage 1 joint | `exp37` | Stage 1 final 19550 | 1 | 2288.441 M | 1285.918 M | 77.003 M | 49.9381 | 2.9983 | 2.6749 | 26.8646 | [JSON](outputs_loc/benchmark_v2/exp37/checkpoint-19550/test/seed_42/benchmark_csgo_v2_loc.json) |
| UniLIP · Stage 2 loc-only | `exp37_s2_loc` | local 6000 | 8 | 2269.423 M | 1285.918 M | 55.689 M | 46.3411 | 2.8749 | 2.5172 | 25.1531 | [JSON](outputs_loc/benchmark_v2/exp37_s2_loc/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json) |
| UniLIP · Stage 2 joint | `exp37_s2` | local 6000 | 8 | 2288.441 M | 1285.918 M | 77.003 M | 47.1732 | 2.8362 | 2.6514 | 26.3147 | [JSON](outputs_loc/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json) |
| UniLIP · Stage 2 P4＋U2 | `exp37_s2_both` | local 6000 | 8 | 2288.441 M | 1285.918 M | 77.003 M | 46.7915 | 2.7853 | 2.7759 | 26.0181 | [JSON](outputs_loc/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json) |
| UniLIP · Stage 2 P3＋U2 | `exp37_s2_both_b1` | local 6000 | 32 | 2288.441 M | 1285.918 M | 77.003 M | 47.5643 | 2.8302 | 2.6703 | 26.0658 | [JSON](outputs_loc/benchmark_v2/exp37_s2_both_b1/checkpoint-6000/test/seed_42/infer_bs32/loc_metrics.json) |

#### 原 batch1 结果（历史留存）

| Method／训练方式 | Experiment | Checkpoint | 推理 batch | Total params | Active params | Trainable params | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 来源 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UniLIP · Stage 2 loc-only | `exp37_s2_loc` | local 6000 | 1 | 2269.423 M | 1285.918 M | 55.689 M | 46.3528 | 2.8725 | 2.5183 | 25.1111 | [JSON](outputs_loc/benchmark_v2/exp37_s2_loc/checkpoint-6000/test/seed_42/benchmark_csgo_v2_loc.json) |
| UniLIP · Stage 2 joint | `exp37_s2` | local 6000 | 1 | 2288.441 M | 1285.918 M | 77.003 M | 47.1936 | 2.8386 | 2.6494 | 26.3374 | [JSON](outputs_loc/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/benchmark_csgo_v2_loc.json) |

### exp37 离散生成（最终test）

| Method／训练方式 | Experiment | Checkpoint | 推理 batch | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 来源 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OmniGen2（原表OmniGen） | `exp32gen_aligned` | final 19500 | 16（初始） | 7805.604 M | 7136.920 M | 5.107 M | 11.9736 | 0.6915 | 0.5057 | 83.4449 | [JSON](../OmniGen2/outputs/csgo_seen10_exp32gen_aligned/OmniGen2/seed_42/evaluation/late/discrete/summary_equal_map.json) |
| ControlAR | `exp32gen_aligned_peft` | final 19500 | 16 | 910.198 M | 859.926 M | 69.317 M | 13.3055 | 0.6230 | 0.5103 | 30.2869 | [JSON](../ControlAR/outputs/csgo_seen10_exp32gen_aligned_peft/ControlAR/seed_42/evaluation/late/inference_seed_42/discrete/summary_equal_map.json) |
| Lumina-Accessory | `--` | 原表未注明 | 未记录 | 5710.672 M§ | 5710.670 M§ | 227.964 M§ | 12.823 | 0.643 | 0.511 | 31.098 | [原表](#离散生成)（产物待绑定§） |
| Show-o2 | `--`‡ | 19500（候选‡） | 未记录 | 3263.077 M‡ | 3030.113 M‡ | 336.481 M‡ | 14.5386 | 0.5910 | 0.5289 | 37.4297 | [候选JSON‡](../Show-o/outputs/csgo_benchmark_v2_aligned/csgo_seen10_exp32gen_aligned_v2_final/seed_42/evaluations/late/discrete/summary_equal_map.json) |
| Puffin | `exp32gen_aligned` | 原表未注明 | 未记录 | 4560.858 M§ | 3898.988 M§ | 88.824 M§ | 14.3186 | 0.5924 | 0.5357 | 33.2448 | [原表](#离散生成)（产物待绑定§） |
| UniLIP full-head · gen-only | `exp31_gen` | final 19500 | 1（配置） | 1785.802 M | 1645.039 M | 683.519 M | 14.5605 | 0.5888 | 0.5258 | 28.6290 | [JSON](outputs_eval/benchmark_v2/exp31_gen/seen/discrete/summary.json) |
| UniLIP full-head · joint | `exp31_1` | final 19500 | 1（配置） | 1785.802 M | 1645.039 M | 1135.684 M | 14.9542 | 0.5725 | 0.5375 | 27.9493 | [JSON](outputs_eval/benchmark_v2/exp31_1/seen/discrete/summary.json) |
| UniLIP full-head · joint＋aux | `exp31` | final 19550 | 1（配置） | 1785.802 M | 1645.039 M | 1135.684 M | 14.6271 | 0.5847 | 0.5294 | 28.3035 | [JSON](outputs_eval/benchmark_v2/exp31/seen/discrete/summary.json) |
| UniLIP LoRA · gen-only | `exp32_gen` | final 19500 | 1（配置） | 1822.416 M | 1681.653 M | 38.910 M | 13.9649 | 0.6106 | 0.5163 | 35.5836 | [JSON](outputs_eval/benchmark_v2/exp32_gen/seen/discrete/summary.json) |
| UniLIP LoRA · joint | `exp32` | final 19550 | 1（配置） | 1822.416 M | 1681.653 M | 77.003 M | 14.1528 | 0.6035 | 0.5229 | 36.6005 | [JSON](outputs_eval/benchmark_v2/exp32/seen/discrete/summary.json) |
| UniLIP LoRA · joint＋aux | `exp32_1` | final 19550 | 1（配置） | 1822.416 M | 1681.653 M | 77.003 M | 13.9070 | 0.6146 | 0.5151 | 36.2159 | [JSON](outputs_eval/benchmark_v2/exp32_1/seen/discrete/summary.json) |
| UniLIP · Stage 1 joint | `exp37` | Stage 1 final 19550 | 1 | 1822.416 M | 1681.653 M | 77.003 M | 13.6163 | 0.6272 | 0.5078 | 36.3942 | [JSON](outputs_eval/benchmark_v2/exp37/checkpoint-19550/test/seed_42/discrete/summary.json) |
| UniLIP · Stage 2 gen-only | `exp37_s2_gen` | local 6000 | 8 | 1822.416 M | 1681.653 M | 38.910 M | 14.1692 | 0.6025 | 0.5217 | 34.3244 | [JSON](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/infer_bs8/discrete/summary.json)；CPU评测；推理batch8 |
| UniLIP · Stage 2 joint | `exp37_s2` | local 6000 | 8 | 1822.416 M | 1681.653 M | 77.003 M | 13.7476 | 0.6230 | 0.5099 | 35.8618 | [JSON](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/infer_bs8/discrete/summary.json) |
| UniLIP · Stage 2 P4＋U2 | `exp37_s2_both` | local 6000 | 8 | 1822.416 M | 1681.653 M | 77.003 M | 13.7378 | 0.6238 | 0.5092 | 35.8402 | [JSON](outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/discrete/summary.json) |
| UniLIP · Stage 2 P3＋U2 | `exp37_s2_both_b1` | local 6000 | 16 | 1822.416 M | 1681.653 M | 77.003 M | 13.7606 | 0.6230 | 0.5101 | 35.6276 | [JSON](outputs_eval/benchmark_v2/exp37_s2_both_b1/checkpoint-6000/test/seed_42/infer_bs16/discrete/summary.json) |

#### 原 batch1 结果（历史留存）

| Method／训练方式 | Experiment | Checkpoint | 推理 batch | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 来源 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UniLIP · Stage 2 gen-only | `exp37_s2_gen` | local 6000 | 1 | 1822.416 M | 1681.653 M | 38.910 M | 14.1668 | 0.6029 | 0.5217 | 34.0804 | [JSON](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/discrete/summary.json) |
| UniLIP · Stage 2 joint | `exp37_s2` | local 6000 | 1 | 1822.416 M | 1681.653 M | 77.003 M | 13.7481 | 0.6229 | 0.5099 | 35.8037 | [JSON](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/discrete/summary.json) |

### exp37 连续生成（最终test）

| Method／训练方式 | Experiment | Checkpoint | 推理 batch | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | T-Warp↓ | T-Diff↓ | FVD↓ | 来源 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OmniGen2（原表OmniGen） | `exp32gen_aligned` | final 19500 | 16（初始） | 7805.604 M | 7136.920 M | 5.107 M | 12.7187 | 0.6993 | 24.9154 | 33.0017 | 1631.3874 | [JSON](../OmniGen2/outputs/csgo_seen10_exp32gen_aligned/OmniGen2/seed_42/evaluation/late/continuous/summary_equal_map.json) |
| ControlAR | `exp32gen_aligned_peft` | final 19500 | 16 | 910.198 M | 859.926 M | 69.317 M | 13.9832 | 0.6157 | 39.2998 | 45.1919 | 951.4971 | [JSON](../ControlAR/outputs/csgo_seen10_exp32gen_aligned_peft/ControlAR/seed_42/evaluation/late/inference_seed_42/continuous/summary_equal_map.json) |
| Lumina-Accessory | `--` | 原表未注明 | 未记录 | 5710.672 M§ | 5710.670 M§ | 227.964 M§ | 13.722 | 0.634 | 38.103 | 44.423 | 932.545 | [原表](#连续生成)（产物待绑定§） |
| Show-o2 | `--`‡ | 19500（候选‡） | 未记录 | 3263.077 M‡ | 3030.113 M‡ | 336.481 M‡ | 15.4463 | 0.5765 | 32.7094 | 39.0428 | 887.3028 | [候选JSON‡](../Show-o/outputs/csgo_benchmark_v2_aligned/csgo_seen10_exp32gen_aligned_v2_final/seed_42/evaluations/late/continuous/summary_equal_map.json) |
| Puffin | `exp32gen_aligned` | 原表未注明 | 未记录 | 4560.858 M§ | 3898.988 M§ | 88.824 M§ | 15.4657 | 0.5679 | 28.1975 | 35.2320 | 844.2427 | [原表](#连续生成)（产物待绑定§） |
| UniLIP full-head · gen-only | `exp31_gen` | final 19500 | 1（配置） | 1785.802 M | 1645.039 M | 683.519 M | 15.1641 | 0.5622 | 31.4090 | 38.0330 | 737.1975 | [JSON](outputs_eval/benchmark_v2/exp31_gen/seen/continuous/summary.json) |
| UniLIP full-head · joint | `exp31_1` | final 19500 | 1（配置） | 1785.802 M | 1645.039 M | 1135.684 M | 15.5707 | 0.5446 | 28.8793 | 35.6882 | 674.1797 | [JSON](outputs_eval/benchmark_v2/exp31_1/seen/continuous/summary.json) |
| UniLIP full-head · joint＋aux | `exp31` | final 19550 | 1（配置） | 1785.802 M | 1645.039 M | 1135.684 M | 15.1897 | 0.5592 | 31.0365 | 37.6569 | 727.5069 | [JSON](outputs_eval/benchmark_v2/exp31/seen/continuous/summary.json) |
| UniLIP LoRA · gen-only | `exp32_gen` | final 19500 | 1（配置） | 1822.416 M | 1681.653 M | 38.910 M | 14.3418 | 0.5929 | 35.6983 | 41.8670 | 848.0710 | [JSON](outputs_eval/benchmark_v2/exp32_gen/seen/continuous/summary.json) |
| UniLIP LoRA · joint | `exp32` | final 19550 | 1（配置） | 1822.416 M | 1681.653 M | 77.003 M | 14.5725 | 0.5826 | 34.0925 | 40.3596 | 848.4586 | [JSON](outputs_eval/benchmark_v2/exp32/seen/continuous/summary.json) |
| UniLIP LoRA · joint＋aux | `exp32_1` | final 19550 | 1（配置） | 1822.416 M | 1681.653 M | 77.003 M | 14.2285 | 0.5981 | 36.1777 | 42.2848 | 891.6025 | [JSON](outputs_eval/benchmark_v2/exp32_1/seen/continuous/summary.json) |
| UniLIP · Stage 1 joint | `exp37` | Stage 1 final 19550 | 1 | 1822.416 M | 1681.653 M | 77.003 M | 13.8818 | 0.6171 | 37.9583 | 43.9776 | 943.6128 | [JSON](outputs_eval/benchmark_v2/exp37/checkpoint-19550/test/seed_42/continuous/summary.json) |
| UniLIP · Stage 2 gen-only | `exp37_s2_gen` | local 6000 | 8 | 1822.416 M | 1681.653 M | 38.910 M | 14.6329 | 0.5814 | 34.4580 | 40.6634 | 822.6760 | [JSON](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/infer_bs8/continuous/summary.json)；CUDA评测；推理batch8 |
| UniLIP · Stage 2 joint | `exp37_s2` | local 6000 | 8 | 1822.416 M | 1681.653 M | 77.003 M | 14.0769 | 0.6085 | 37.0982 | 43.1798 | 928.6481 | [JSON](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/infer_bs8/continuous/summary.json) |
| UniLIP · Stage 2 P4＋U2 | `exp37_s2_both` | local 6000 | 8 | 1822.416 M | 1681.653 M | 77.003 M | 14.0884 | 0.6082 | 37.1505 | 43.2095 | 922.6230 | [JSON](outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/continuous/summary.json) |
| UniLIP · Stage 2 P3＋U2 | `exp37_s2_both_b1` | local 6000 | 16 | 1822.416 M | 1681.653 M | 77.003 M | 14.1008 | 0.6078 | 37.1682 | 43.2016 | 906.5443 | [JSON](outputs_eval/benchmark_v2/exp37_s2_both_b1/checkpoint-6000/test/seed_42/infer_bs16/continuous/summary.json) |

#### 原 batch1 结果（历史留存）

| Method／训练方式 | Experiment | Checkpoint | 推理 batch | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | T-Warp↓ | T-Diff↓ | FVD↓ | 来源 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UniLIP · Stage 2 gen-only | `exp37_s2_gen` | local 6000 | 1 | 1822.416 M | 1681.653 M | 38.910 M | 14.6232 | 0.5814 | 34.5009 | 40.6948 | 843.4360 | [JSON](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/continuous/summary.json) |
| UniLIP · Stage 2 joint | `exp37_s2` | local 6000 | 1 | 1822.416 M | 1681.653 M | 77.003 M | 14.0911 | 0.6077 | 37.1104 | 43.1822 | 922.5427 | [JSON](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/continuous/summary.json) |

<a id="exp31-exp37-parameter-audit-20261109"></a>

### exp31／exp37 参数量核对（2026-11-09）

本次补齐本节三张最终test主表及历史batch1表中30行的三列参数量，单位仍为 M = 10⁶，保留三位小数；下表保留精确整数。核对读取了各实验 `model.safetensors` 的张量名称／形状、训练日志逐项参数表、冻结／LoRA配置及实际前向代码，没有加载权重主体或启动训练、推理、评测。推理batch、数据split、checkpoint步数不改变同一结构的参数量；本次不调整已有评测指标。

| 实验与任务 | Total（个） | Active（个） | Trainable（个） | 权重／训练记录依据 |
|---|---:|---:|---:|---|
| `exp31_loc`；Localization | 2,237,966,083 | 1,254,461,088 | 452,164,384 | [权重](outputs/csgo_1b/exp31_loc/model.safetensors)、[逐项训练日志](logs/csgo_1b/exp31_loc/train_20260831_180447/train.log) |
| `exp31_gen`；离散／连续生成 | 1,785,801,699 | 1,645,039,011 | 683,519,136 | [权重](outputs/csgo_1b/exp31_gen/model.safetensors)、[逐项训练日志](logs/csgo_1b/exp31_gen/train_20260831_175920/train.log) |
| `exp31_1`／`exp31`；Localization | 2,237,966,083 | 1,254,461,088 | 1,135,683,520 | [exp31_1权重](outputs/csgo_1b/exp31_1/model.safetensors)、[exp31权重](outputs/csgo_1b/exp31/model.safetensors)、[无辅助训练日志](logs/csgo_1b/exp31_1/train_20260907_220547/train.log)、[辅助训练日志](logs/csgo_1b/exp31/train_20260831_172736/train.log) |
| `exp31_1`／`exp31`；离散／连续生成 | 1,785,801,699 | 1,645,039,011 | 1,135,683,520 | 同上；生成入口不构建定位私有模块 |
| `exp37`／`exp37_s2`／`exp37_s2_both`；Localization | 2,288,441,091 | 1,285,918,368 | 77,002,784 | 各自[Stage1权重](outputs/csgo_1b/exp37/model.safetensors)、[Stage2 joint权重](outputs/csgo_1b/exp37_s2/model.safetensors)、[Stage2 both权重](outputs/csgo_1b/exp37_s2_both/model.safetensors)；[both逐项训练日志](logs/csgo_1b/exp37_s2_both/train_20261103_142251/train.log)交叉核验 |
| `exp37`／`exp37_s2`／`exp37_s2_both`；离散／连续生成 | 1,822,415,843 | 1,681,653,155 | 77,002,784 | 同上；各自实际权重和resolved配置独立核对，未仅按旧exp32结果填入 |
| `exp37_s2_loc`；Localization | 2,269,423,363 | 1,285,918,368 | 55,688,992 | [权重](outputs/csgo_1b/exp37_s2_loc/model.safetensors)、[逐项训练日志](logs/csgo_1b/exp37_s2_loc/train_20261101_142105/train.log) |
| `exp37_s2_gen`；离散／连续生成 | 1,822,415,843 | 1,681,653,155 | 38,910,208 | [权重](outputs/csgo_1b/exp37_s2_gen/model.safetensors)、[逐项训练日志](logs/csgo_1b/exp37_s2_gen/train_20261101_142058/train.log) |
| `exp37_s2_both_b1`；Localization | 2,288,441,091 | 1,285,918,368 | 77,002,784 | [最终权重](outputs/csgo_1b/exp37_s2_both_b1/checkpoint-6000/model.safetensors)、[逐项训练日志](logs/csgo_1b/exp37_s2_both_b1/train_20261103_152114/train.log)、[定位清单](outputs_loc/benchmark_v2/exp37_s2_both_b1/checkpoint-6000/test/seed_42/infer_bs32/inference_manifest.json)；头部张量形状和训练参数表核对 |
| `exp37_s2_both_b1`；离散／连续生成 | 1,822,415,843 | 1,681,653,155 | 77,002,784 | 同一最终权重／训练日志；[生成清单](outputs_eval/benchmark_v2/exp37_s2_both_b1/checkpoint-6000/test/seed_42/infer_bs16/discrete/inference_manifest.json)；生成入口不构建定位私有模块 |

**Total／Active 的计算。** [定位入口](eval_csgo_loc.py)会初始化定位头并保留生成模块；[生成入口](eval_csgo.py)不调用 `initialize_localization_modules`，实际不加载定位私有模块。[模型前向](unilip/model/language_model/unified_unilip.py)据此逐模块核对：

- `exp31*` 完整checkpoint均为 **2,237,966,083**；定位Active = 视觉塔304,012,288 + 多模态投影4,482,816 + LLM 493,801,600 + 定位私有模块452,164,384 = **1,254,461,088**。生成Total = 2,237,966,083 − 452,164,384 = **1,785,801,699**。
- `exp37`／`exp37_s2`／`exp37_s2_both` 完整checkpoint均为 **2,288,441,091**，`exp37_s2_loc` 为 **2,269,423,363**，`exp37_s2_gen` 为 **2,274,580,227**。定位Active = 304,012,288 + 4,482,816 + LLM（含LoRA）511,398,016 + 定位私有模块（含LoRA）466,025,248 = **1,285,918,368**。单任务定位未注入生成头LoRA，因此Total比joint少19,017,728，但定位Active相同；所有适用的生成行均只加载生成结构，Total为 **1,822,415,843**。
- 两系列生成Active均从各自生成Total扣除未调用的 `lm_head` **135,903,488** 和 `vae_decoder.down_blocks/down_mlp` **4,859,200**；[VAE实现](unilip/model/vae_modules.py)的实际decoder为159,223,171，无需再计一个独立完整VAE。定位Active也不含语言输出头或生成分支。
- `lm_head` 与输入embedding未共享参数，故在Total中分别计数；checkpoint中的参数项与训练 `named_parameters` 对应，buffer不计。非LoRA分支的[训练日志计数代码](train_csgo.py)只遍历 `model.get_model()`，日志Total **2,102,062,595** 漏掉外层 `lm_head`；补上135,903,488才等于exp31的完整模型Total。该头冻结，不影响日志Trainable。LoRA推理未merge／卸载，适配器仍计入Total／Active。

**Trainable 的计算。** 按当前训练阶段优化器覆盖且损失可达的参数张量并集统计，不累计两阶段的参数次数，也不把联合行按任务拆分：

- `exp31*` 冻结共享视觉／语言主干，full-head生成可训练参数 = DiT 591,747,872 + LLM connector 89,475,200 + projector 2,066,688 + latent queries 229,376 = **683,519,136**；定位私有可训练参数 **452,164,384**；联合为两者之和 **1,135,683,520**。四组逐项训练日志的可训练项数分别为loc 221、gen 624、joint 845、joint＋aux 845，与权重形状汇总一致。
- `exp37*` 的共享LLM LoRA为 **17,596,416**；生成私有可训练部分为 **21,313,792**（connector LoRA 2,199,552 + DiT LoRA 16,818,176 + projector 2,066,688 + queries 229,376），定位私有可训练部分为 **38,092,576**（action DiT LoRA 13,860,864及其余定位投影、归一化、输入输出与时间模块24,231,712）。因此gen-only为 **38,910,208**、loc-only为 **55,688,992**、joint为 **77,002,784**；共享部分只计一次。实际适配器rank为LLM／DiT／action DiT 32、connector 16，已与各checkpoint形状核对。
- perception／aux-loc复用现有模块，未引入额外可训练参数或推理模块；临时冻结辅助分支不改变定位主损失可更新的参数并集。`exp37_s2_both_b1` 的训练日志记录Total 2,288,441,091、Trainable 77,002,784，Stage1 严格加载 2,830 个张量、无 missing／unexpected；最终 local6000 权重头部为 2,830 个张量、合计 2,288,441,091 个参数，声明数据大小与文件大小一致。root／local4000／local6000 的头部名称和形状与 B2 一致；本核对未重算整份权重的 SHA。B1 与 B2 的配方差异为 perception attention true／false；实际训练均为双卡 micro4、GAS16、有效 batch128，B1 的 save_steps=1000、B2=2000。两者最终评测推理 batch 不同，不能据 test 差值单独判断 attention 效果。本次静态核查未发现上述各配方存在需要从日志Trainable中扣除的整块损失断开参数张量。
