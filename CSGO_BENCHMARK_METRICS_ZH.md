# UniLIP-CSGO Benchmark Metric 中文说明

本文档是 UniLIP-CSGO Benchmark 的指标口径说明，用于回答两个问题：

- 论文主表和附录应报告哪些质量指标；
- 实验结果 JSON 中每个指标的含义、计算方法和主要测量目标。

本文档描述的是当前代码实现。对历史 JSON 字段的解释不等于建议继续报告该指标。
定位、离散生成、连续生成和多地图汇总的实现源分别是
[`eval_csgo_loc.py`](eval_csgo_loc.py)、[`benchmark_csgo_v1.py`](benchmark_csgo_v1.py)、
[`benchmark_csgo_v1_conti.py`](benchmark_csgo_v1_conti.py) 和
[`scripts/aggregate_csgo_benchmark_v2_metrics.py`](scripts/aggregate_csgo_benchmark_v2_metrics.py)。

## 正式报告的指标集

| 任务 | 主表中的质量 metric | 附录中的 metric（包含主表 metric） |
| --- | --- | --- |
| 定位 | `XY_Dist↓`、`Z_Dist↓`、`Pitch_Dist↓`、`Yaw_Dist↓` | `XY_Dist↓`、`Z_Dist↓`、`Pitch_Dist↓`、`Yaw_Dist↓`、`Norm_L2_5D↓`、`L1_X↓`、`L1_Y↓` |
| 离散生成 | `PSNR↑`、`SSIM↑`、`LPIPS↓`、`Boundary_F1↑`、`FID↓` | `PSNR↑`、`SSIM↑`、`LPIPS↓`、`Boundary_F1↑`、`FID↓`、`Pixel_MAE_255↓`、`CLIP↑` |
| 连续生成 | `PSNR↑`、`SSIM↑`、`LPIPS↓`、`Temporal_Warping_Error↓`、`Temporal_Difference_Error↓`、`FVD↓` | `PSNR↑`、`SSIM↑`、`LPIPS↓`、`Temporal_Warping_Error↓`、`Temporal_Difference_Error↓`、`FVD↓`、`Boundary_F1↑`、`Optical_Flow_EPE↓`、`Pixel_MAE_255↓` |

`Coverage_GT`、`Coverage_Pred`、`Common_Count`、`Track_Count`、`Seq_Frame_Count` 和 `fvd_clip_count` 是必须保留的完整性/协议检查，但不计入上表的质量 metric 数量，也不参与模型质量排名。

联合模型与对应的 single-task control 必须使用相同指标。Seen-10、CrossMap-4 zero-shot、few-shot 和 Seen-retention 也必须使用同一指标集。所有结果先按 map 报告，再计算 equal-map macro。

所有 `Locator_*` 指标均不进入正式主表或附录。历史 JSON 仍可能包含这些字段，因此本文档在后文保留它们的解释。

## JSON 结构和汇总字段

| JSON 字段 | 含义和计算方法 | 性质 |
| --- | --- | --- |
| `metrics_ordered` | 单张地图的 metric，按评测脚本预设顺序保存。 | 单地图结果 |
| `per_map` | 聚合 JSON 中每张地图各自的 metric 集合。 | 单地图结果 |
| `metrics_macro_map` | 先分别计算每张地图，再对地图做等权算术平均。每张地图权重相同，不按样本数加权。 | 主要汇总结果 |
| `metrics_support_selection` | 多个 support seed 的均值、标准差或置信区间汇总。当前 `exp33*`/`exp34*` 正式协议只使用 `seed_0`，不能由此声称 support-selection 稳定性。 | 可选多 seed 分析 |
| `ckpt_path` | 产生结果的 checkpoint 路径。 | 溯源信息，不是质量 metric |

### Map-models 汇总

Benchmark v2 的 `map-models` 是为 map-specific few-shot 实验增加的等权
地图汇总方式。它从 `--input_pattern` 中的 `{map}` 占位符读取每张地图的
一个结果文件；这些文件可以来自四个不同的 checkpoint。它先保留每张图的
metric，再对 `cs_office`、`de_golden`、`de_palacio`、`de_vertigo` 四张
CrossMap-4 地图做等权平均。

这三种汇总不能混用：

| 子命令 | 地图结果来自 | 适用场景 |
| --- | --- | --- |
| `maps` | 同一个 checkpoint，在一个输入目录中产生全部协议地图结果 | `exp33`/`exp34` 等单模型跨地图评测，或单个 map-specific 模型的 Seen-retention |
| `seeds` | 同一模型路线的不同 support selection | 比较 support seed 的波动；不是独立训练 seed |
| `map-models` | 每张地图一个专门 checkpoint | `exp35*`/`exp36*` 的 CrossMap-4 family macro |

因此，单个 `exp35_<map>` 或 `exp36_<map>` 的 CrossMap metric 必须先直接
报告该目标地图，不能调用要求一个 checkpoint 覆盖全部协议地图的 `maps`
汇总。四个 map-specific 结果的 family-level macro 才使用新增的命令契约：

```bash
python scripts/aggregate_csgo_benchmark_v2_metrics.py map-models \
  --manifest data/csgo_benchmark_v2/benchmark_manifest.json \
  --split crossmap_query_test \
  --kind discrete \
  --input_pattern 'outputs_eval/benchmark_v2/exp35_{map}/shot_100/seed_0/discrete/benchmark_csgo_v2_{map}.json' \
  --output outputs_eval/benchmark_v2/exp35/shot_100/seed_0/map_models/benchmark_v2_discrete_crossmap_query_test.json
```

`--kind continuous` 使用同一结构下的
`continuous/benchmark_csgo_v2_conti_{map}.json`；`--kind localization` 使用
`outputs_loc/benchmark_v2/exp35_{map}/shot_100/seed_0/benchmark_csgo_v2_loc.json`。
合法路线必须按任务区分：

| metric kind | 合法实验路线 | 说明 |
| --- | --- | --- |
| `discrete`、`continuous` | `exp35`、`exp35_gen`、`exp36`、`exp36_gen` | 联合模型或 gen-only 模型的生成结果 |
| `localization` | `exp35`、`exp35_loc`、`exp36`、`exp36_loc` | 联合模型或 loc-only 模型的定位结果 |

因此不能对 `exp35_loc`/`exp36_loc` 做离散或连续生成汇总，也不能对
`exp35_gen`/`exp36_gen` 做定位汇总。输出必须保留 map 到 checkpoint 的
映射，避免把 map-models macro 误解为单个模型在四张地图上的泛化结果。

完整的 exp35/36 三种任务路线命令示例见
[`CSGO_BENCHMARK_V2_MAP_SPECIFIC_FEWSHOT.md`](CSGO_BENCHMARK_V2_MAP_SPECIFIC_FEWSHOT.md)。

## 定位任务 JSON metric

预测和 GT pose 的顺序均为 `[x, y, z, pitch, yaw]`。X/Y 通过除以 1024 归一化，Z 使用每张地图冻结的 `z_min`/`z_max` 归一化，角度通过除以 360° 归一化。物理空间中 X/Y/Z 的单位是地图/游戏坐标，不是米。

| Metric | 含义和计算方法 | 着重测量的点 | 趋势 |
| --- | --- | --- | --- |
| `XY_Dist` | 对每个样本计算 `sqrt((x_pred-x_gt)^2+(y_pred-y_gt)^2)`，再对所有样本求平均。 | 玩家在地图平面上的定位准确度。 | ↓ |
| `Z_Dist` | 预测高度和真实高度绝对差的平均值。 | 楼层、高台、坡道等垂直位置是否正确。 | ↓ |
| `Pitch_Dist` | 预测 pitch 与真实 pitch 的角度绝对差的平均值，单位为度。Pitch 不做循环 wrap。 | 镜头向上或向下看的角度。 | ↓ |
| `Yaw_Dist` | 按设计应计算 yaw 的最短圆周角距离，例如 359° 和 1° 的距离是 2°，然后对样本求平均。 | 镜头水平朝向是否正确。 | ↓ |
| `L1_X` | X 坐标绝对误差的平均值。 | 平面定位中的 X 分量误差。 | ↓ |
| `L1_Y` | Y 坐标绝对误差的平均值。 | 平面定位中的 Y 分量误差。 | ↓ |
| `L1_Z` | 与 `Z_Dist` 完全相同。 | 高度误差。 | ↓ |
| `L1_Pitch` | 与 `Pitch_Dist` 完全相同。 | 俯仰角误差。 | ↓ |
| `L1_Yaw` | 与 `Yaw_Dist` 完全相同。 | 水平朝向误差。 | ↓ |
| `L2_XY` | 与 `XY_Dist` 完全相同，只是名称不同。 | 地图平面定位误差。 | ↓ |
| `Norm_L2_XY` | 先在归一化空间计算 XY 欧氏距离，再求平均。本项目中基本等于 `XY_Dist/1024`。 | 消除 XY 数值尺度后的平面误差。 | ↓ |
| `Norm_L2_5D` | 对每个样本的归一化五维误差计算欧氏距离 `sqrt(dx^2+dy^2+dz^2+dpitch^2+dyaw^2)`，再求平均。 | 用一个数概括整体 5DoF 定位误差。 | ↓ |
| `Norm_MSE_5D` | 对归一化五维中的每个误差平方，然后对样本和五个维度一起求平均。 | 对较大的归一化误差施加更强惩罚。 | ↓ |
| `Norm_SmoothL1_5D` | 对归一化误差计算 Smooth L1/Huber loss：小误差使用平方惩罚，大误差近似线性惩罚。 | 在关注大误差的同时减少极端离群样本的影响。 | ↓ |
| `L2_5D` | 在物理数值空间直接计算 `[X,Y,Z,pitch,yaw]` 的五维欧氏距离。由于混合了地图坐标和角度，数值不具有统一物理量纲。 | 尝试用一个数表示整体误差。 | ↓ |
| `MSE_Loss_5D` | 在物理数值空间对五个分量计算均方误差。 | 强调特别大的定位错误，但同样混合不同量纲。 | ↓ |
| `SmoothL1_Loss_5D` | 在物理数值空间对五个分量计算 Smooth L1。 | 比 MSE 更不容易被极端误差支配，但同样混合不同量纲。 | ↓ |
| `TrainAligned_LocLoss` | 按训练配置分别计算 XY、Z 和角度 Smooth L1，再乘以对应 loss weight 后相加。 | 检查评测结果与当前训练目标是否一致。 | ↓ |
| `TrainAligned_TotalLoss` | `TrainAligned_LocLoss * alpha_loc_loss`。 | 定位 loss 在总训练目标中的加权值。 | ↓ |

## 离散生成任务 JSON metric

离散生成将每张生成 FPS 图与同一 pose 对应的真实 FPS 图进行比较。

| Metric | 含义和计算方法 | 着重测量的点 | 趋势 |
| --- | --- | --- | --- |
| `Coverage_GT` | `成功匹配的预测数 / 应评测的 GT 数`。例如应生成 2000 张但只有 1900 张，则结果为 0.95。 | 是否漏生成样本。 | 必须为 1 |
| `Coverage_Pred` | `成功匹配的预测数 / 预测目录中的总图像数`。 | 是否产生多余、命名错误或不属于当前 split 的图片。 | 必须为 1 |
| `Common_Count` | GT 和预测中成功按文件名匹配的样本数。 | 实际参与质量计算的样本量。 | 应等于协议值 |
| `PSNR` | 先计算生成图与 GT 的像素均方误差 MSE，再计算 `10*log10(1/MSE)` 并对样本求平均。 | 像素级还原程度，尤其是颜色和亮度误差。 | ↑ |
| `SSIM` | 比较局部区域的亮度、对比度和结构，再对图像和样本求平均。 | 墙体、通道、物体等整体结构是否相似。 | ↑ |
| `LPIPS` | 将生成图和 GT 输入预训练 AlexNet，比较深层特征之间的距离。 | 人眼感知上的相似度，而不是逐像素完全一致。 | ↓ |
| `Boundary_F1` | 使用 Sobel 算子分别找出 GT 和生成图中的强边缘，允许默认约 2 像素的位置偏差，再计算边缘 precision 和 recall 的调和平均。 | 墙边、门框、箱子和建筑轮廓是否对齐。 | ↑ |
| `Boundary_Precision` | 生成图检测到的边缘中，有多少能在 GT 边缘附近找到匹配。 | 是否生成了多余或错误轮廓。 | ↑ |
| `Boundary_Recall` | GT 边缘中，有多少能在生成图边缘附近找到匹配。 | 是否遗漏了真实轮廓。 | ↑ |
| `Boundary_GT_Edge_Ratio` | GT 图中被判定为边缘的像素比例。 | 检查 GT 边缘提取结果是否异常。 | 无固定方向 |
| `Boundary_Pred_Edge_Ratio` | 生成图中被判定为边缘的像素比例。 | 检查生成图是否过度锐化或缺乏边缘。 | 应接近 GT |
| `Pixel_MAE_255` | 对每个 RGB 通道计算 `abs(pred-gt)`，再对所有通道、像素和图像求平均，数值范围按 0–255 表示。 | 最直观的平均像素偏差。 | ↓ |
| `Pixel_Exact_Acc` | 一个像素的 R、G、B 三个通道全部与 GT 完全相同才计为正确。 | 完全一致的像素比例。 | ↑ |
| `Pixel_Within_1_Acc` | 一个像素的三个颜色通道与 GT 的误差都不超过 1 才计为正确。 | 近乎完全一致的像素比例。 | ↑ |
| `FID` | 分别提取 GT 和生成图的 Inception 特征，用均值和协方差描述两个特征分布，再计算两个分布之间的 Fréchet 距离。 | 整批生成图的整体真实性和分布覆盖，不是单张配对准确性。 | ↓ |
| `IS` | 用 Inception 分类器预测生成图类别；单张图类别越明确、整批图类别越多样，分数越高。该指标不使用 GT。 | 生成图的可分类性和多样性。 | ↑ |
| `CLIP` | 分别提取 GT 和生成图的 CLIP 图像特征，计算两者的余弦相似度，再对样本求平均。当前实现不是文本—图像 CLIP Score。 | 高层语义是否相似，例如是否大致为相同场景。 | ↑ |
| `Aesthetic` | 用 CLIP ViT-L/14 提取生成图特征，再由预训练 aesthetic MLP 给出美学分数。该指标只看生成图，不看 GT。 | 图像是否符合通用的美学偏好。 | ↑ |

### 历史 JSON 中的外部 locator metric

以下字段由冻结的 CSGOSquare 外部 locator 计算：将生成 FPS 图和 radar 图输入 locator，让它反推 pose，再与生成时要求的目标 pose 比较。它们测到的是“外部 locator 如何理解生成图”，不等于生成图的真实 pose 误差。

| Metric | 含义和计算方法 | 着重测量的点 | 趋势 |
| --- | --- | --- | --- |
| `Locator_XY_Dist` | locator 从生成图预测出的 XY 与目标 XY 的平均欧氏距离。 | 生成场景的位置是否符合输入 pose。 | ↓ |
| `Locator_Z_Dist` | locator 预测高度与目标高度的平均绝对差。 | 生成图是否对应正确楼层或高度。 | ↓ |
| `Locator_Pitch_Dist` | locator 预测 pitch 与目标 pitch 的平均角度差。 | 生成镜头的上下朝向。 | ↓ |
| `Locator_Yaw_Dist` | locator 预测 yaw 与目标 yaw 的平均最短圆周角距离。 | 生成镜头的水平朝向。 | ↓ |
| `Locator_Norm_L2_5D` | 在归一化空间把 locator 的 XY、Z、pitch 和 yaw 误差合并成五维欧氏距离。 | 用一个数概括生成图的 pose 条件遵循程度。 | ↓ |

这些 `Locator_*` 字段仅用于解释历史 JSON，不进入当前正式报告的主表或附录。

## 连续生成任务额外 JSON metric

连续生成 JSON 会先包含离散生成的完整 metric 集合，然后增加以下序列字段。Benchmark v2 中每张地图的正式连续 split 包含 20 个 64 帧 clip。

| Metric | 含义和计算方法 | 着重测量的点 | 趋势 |
| --- | --- | --- | --- |
| `Track_Count` | 实际成功组成的连续 clip 数。 | 连续评测轨迹是否完整。 | 应等于协议值 |
| `Seq_Frame_Count` | 所有有效连续 clip 包含的总帧数。 | 实际参与时序评测的帧数。 | 应等于协议值 |
| `Seq-PSNR` | 逐帧计算 PSNR，先在每个 clip 内平均，再对所有 clip 平均。 | 每段轨迹中的逐帧像素还原程度。 | ↑ |
| `Seq-SSIM` | 逐帧计算 SSIM，先在每个 clip 内平均，再跨 clip 平均。 | 连续轨迹中每一帧的结构质量。 | ↑ |
| `Seq-LPIPS` | 逐帧计算 LPIPS，先在每个 clip 内平均，再跨 clip 平均。 | 连续轨迹中每一帧的感知质量。 | ↓ |
| `Seq-MAE` | 逐帧计算 RGB 绝对误差，再按 clip 和地图平均。 | 连续轨迹的平均像素误差。 | ↓ |
| `Seq-Exact-Acc` | 逐帧计算三个 RGB 通道都完全一致的像素比例，再按 clip 平均。 | 连续序列中的完全像素重建率。 | ↑ |
| `Seq-Within_1-Acc` | 逐帧计算三个通道误差都不超过 1 的像素比例，再按 clip 平均。 | 连续序列中的近似完全重建率。 | ↑ |
| `Temporal_Warping_Error` | 从相邻 GT 帧估计光流，用该光流把前一张生成帧 warp 到下一时刻，再与实际的下一张生成帧计算像素 MAE，最后在相邻帧和 clip 间平均。 | 生成内容是否按真实运动方向平滑移动，而不是随机跳动。 | ↓ |
| `Temporal_Difference_Error` | 先分别计算生成序列和 GT 序列的相邻帧变化量，再计算 `abs((pred_next-pred_prev)-(gt_next-gt_prev))` 的通道/像素平均值。 | 生成序列每一步变化的大小和方向是否与 GT 一致。 | ↓ |
| `Flicker_Score` | 比较生成序列和 GT 序列相邻帧变化强度的差：`abs(abs(pred_next-pred_prev)-abs(gt_next-gt_prev))`。 | 生成视频是否出现过强或过弱的明暗、纹理闪烁。 | ↓ |
| `Optical_Flow_EPE` | 分别用 Farneback 算法估计 GT 和生成序列的光流，对每个像素计算两个二维光流向量的欧氏距离，再求平均。 | 生成视频中的运动方向和运动速度是否正确。 | ↓ |
| `FVD` | 将连续序列切成 16 帧窗口，用预训练 I3D 提取视频特征，再计算 GT 与生成视频特征分布的 Fréchet 距离。当前设置为 stride 16，不跨越 64 帧 clip 边界。 | 整体视频的外观真实性、运动模式和时序分布。 | ↓ |
| `fvd_clip_count` | 实际参与 FVD 计算的 16 帧窗口数。20 个 64 帧 clip、stride 16 时，每张地图通常是 80 个窗口。 | 判断 FVD 样本量和协议覆盖是否完整。 | 应等于协议值 |

## 当前实现的解读注意事项

- 定位主路径中的 yaw 距离当前使用 `min(d, period-d)`。如果预测越出正常范围，可能产生错误甚至负的 yaw 误差；在修正为 modulo/remainder 最短圆周距离前，解读 `Yaw_Dist`、`Norm_L2_5D` 等字段时必须保留这一限制。
- `TrainAligned_LocLoss` 当前启用 circular loss 时同时 wrap pitch 和 yaw，而 Benchmark v2 规定只有 yaw 循环、pitch 不循环。该字段仅适合训练诊断。
- `XY_Dist` 与 `L2_XY` 完全重复；`Z_Dist`、`Pitch_Dist`、`Yaw_Dist` 分别与 `L1_Z`、`L1_Pitch`、`L1_Yaw` 完全重复。
- `L2_5D`、`MSE_Loss_5D` 和 `SmoothL1_Loss_5D` 直接混合地图坐标、高度和角度，不具有统一物理量纲，不用于正式主表。
- `Boundary_F1` 使用每张图各自的 Sobel 边缘分位数阈值，适合表示轮廓对齐，但不能单独代表整体图像质量或清晰度。
- Benchmark v2 的每个 clip 都固定为 64 帧，因此 `Seq-PSNR/SSIM/LPIPS/MAE/Exact/Within_1` 与对应的普通逐帧 metric 几乎相同。它们的区别主要是聚合顺序，不需要同时放入正式结果表。
- FVD 在当前设置下每张地图只有 80 个 16 帧窗口，是有用的视频分布指标，但样本较少，不应单独支撑实验结论。
- `Pixel_Exact_Acc`、`Pixel_Within_1_Acc`、`IS`、`Aesthetic` 仍可作为历史 JSON 字段读取，但不进入当前正式报告。
