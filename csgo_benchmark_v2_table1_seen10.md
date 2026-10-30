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
- 参数量于 **2026-10-19** 按本机实际代码、训练审计、checkpoint 张量形状及推理清单核对。三列统一用 **M = 10⁶ 个参数**，保留三位小数；精确整数和来源见文末“参数量核对依据”。
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

## exp37 新实验身份与 Table 1 待填项（2026-10-30）

本节登记五次新训练的候选身份，协议见 [`CSGO_BENCHMARK_V2_EXP37_SEEN10.md`](CSGO_BENCHMARK_V2_EXP37_SEEN10.md)。**尚未训练、评测或审计参数**，故不把新组替换进上方历史 Table 1 有效数值行，也不借用 `exp32` 的参数量。未来只有在模型结构、任务推理路径、优化器覆盖、完整十图指标和正式 checkpoint 均核对后，才可将相应新行加入正式表格。

| 候选 Experiment | Table 1 任务行 | 固定报告 checkpoint | 当前指标 | 参数量 |
|---|---|---|---|---|
| `exp37` | 定位、离散生成、连续生成 | Stage 1 实际 final，仅参照 | 待执行 | 待审计 |
| `exp37_s2_gen` | 离散生成、连续生成 | Stage 2 local 6000 | 待执行 | 待审计 |
| `exp37_s2_loc` | 定位 | Stage 2 local 6000 | 待执行 | 待审计 |
| `exp37_s2` | 定位、离散生成、连续生成 | Stage 2 local 6000 | 待执行 | 待审计 |
| `exp37_s2_both` | 定位、离散生成、连续生成 | Stage 2 local 6000 | 待执行 | 待审计 |

`exp37_s2_gen`/`loc` 分别来自旧 Seen-10 `exp32_gen@19500`/`exp32_loc@19500` 的模型权重；两个 joint Stage 2 来自同一个新 `exp37` 完整 Stage 1 最终模型。Stage 2 首次初始化重置训练状态，不能将来源 checkpoint 的旧指标冒充新组结果。所有新结果按十图等权宏平均，定位和离散生成 test 各 20,000 条，连续生成 12,800 帧。两阶段联合组须同时披露训练预算；现有资料不足以主张与旧 baseline FLOPs 相等。旧 B1=P3+U2、attention=true；新 both=P4+U2、attention=false，效果尚待实测。
