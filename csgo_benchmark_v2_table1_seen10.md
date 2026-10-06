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
- Total params 表示推理期间加载的全部参数；Active params 表示任务专用前向计算涉及的参数；Trainable params 表示 Seen-10 训练集适配期间更新的参数。
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
| Seen-10 | Localization | X-VLA | `X-VLA-fair-frozen-vl` | -- | -- | -- | 65.969 | 4.162 | 2.908 | 27.748 |
| Seen-10 | Localization | RDT | `--` | -- | -- | -- | 43.251 | 2.783 | 2.306 | 20.779 |
| Seen-10 | Localization | OpenVLA-OFT | `--` | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Localization | $\pi_{0.5}^{\dagger}$ | `pi0.5_exp32_loc_main_frozen_vl` | -- | -- | -- | 53.542 | 3.453 | 1.251 | 25.450 |
| Seen-10 | Localization | pose prediction baseline | `exp32_loc` | -- | -- | -- | 48.961 | 3.006 | 2.970 | 26.047 |
| Seen-10 | Localization | multi-task baseline | `exp32` | -- | -- | -- | 123.194 | 6.38 | 3.352 | 55.449 |
| Seen-10 | Localization | aux. objs. baseline | `exp32_1` | -- | -- | -- | 63.0377 | 3.5604 | 3.0882 | 31.6713 |

## 离散生成

| Setting | Task | Method | Experiment | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | Boundary_F1↑ | FID↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Discrete generation | OmniGen | `--` | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Discrete generation | ControlAR | `csgo_seen10_exp32gen_aligned_peft` | -- | -- | -- | 13.3055 | 0.6230 | 0.5103 | 30.2869 |
| Seen-10 | Discrete generation | Lumina-Accessory | `--` | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Discrete generation | Show-o2 | `--` | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Discrete generation | Puffin | `--` | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Discrete generation | FPV synthesis baseline | `exp32_gen` | -- | -- | -- | 13.965 | 0.6106 | 0.5163 | 35.584 |
| Seen-10 | Discrete generation | multi-task baseline | `exp32` | -- | -- | -- | 14.1528 | 0.6035 | 0.5229 | 36.6005 |
| Seen-10 | Discrete generation | aux. objs. baseline | `exp32_1` | -- | -- | -- | 13.9070 | 0.6145 | 0.5151 | 36.2159 |

## 连续生成

| Setting | Task | Method | Experiment | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | T-Warp↓ | T-Diff↓ | FVD↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Continuous generation | OmniGen | `--` | -- | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Continuous generation | ControlAR | `csgo_seen10_exp32gen_aligned_peft` | -- | -- | -- | 13.9832 | 0.6157 | 39.2998 | 45.1919 | 951.4971 |
| Seen-10 | Continuous generation | Lumina-Accessory | `--` | -- | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Continuous generation | Show-o2 | `--` | -- | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Continuous generation | Puffin | `--` | -- | -- | -- | -- | -- | -- | -- | -- |
| Seen-10 | Continuous generation | FPV synthesis baseline | `exp32_gen` | -- | -- | -- | 14.342 | 0.5929 | 35.698 | 41.867 | 848.071 |
| Seen-10 | Continuous generation | multi-task baseline | `exp32` | -- | -- | -- | 14.5725 | 0.5826 | 34.0925 | 40.3596 | 848.4586 |
| Seen-10 | Continuous generation | aux. objs. baseline | `exp32_1` | -- | -- | -- | 14.2285 | 0.5981 | 36.1777 | 42.2848 | 891.6025 |
