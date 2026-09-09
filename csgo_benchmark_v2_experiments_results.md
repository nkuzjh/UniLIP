


# csgo benchmark v2 实验进度
| 实验版本 | 训练 | 推理 | Metric 计算 |
|---|---|---|---|
| `exp31` | ⏳ Seen-10 训练中，当前评测 checkpoint-6000 | ✅ Seen-10：定位 20,000 + 离散生成 20,000 + 连续生成 12,800 | ✅ Seen-10：定位/离散/连续 metric 与聚合完成 |
| `exp31_loc` | ✅ Seen-10，step 19500 | ✅ Seen-10，20,000 样本 | ✅ Seen-10 定位 metric |
| `exp31_gen` | ✅ Seen-10，step 19500 | ✅ Seen-10：离散 20,000 + 连续 12,800<br>✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ Seen-10 离散/连续<br>✅ CrossMap-4 离散/连续 |
| `exp32` | ⏳ Seen-10 训练中，当前评测 checkpoint-6000 | ✅ Seen-10：定位 20,000 + 离散生成 20,000 + 连续生成 12,800 | ✅ Seen-10：定位/离散/连续 metric 与聚合完成 |
| `exp32_loc` | ✅ Seen-10，step 19500 | ✅ Seen-10，20,000 样本 | ✅ Seen-10 定位 metric |
| `exp32_gen` | ✅ Seen-10，step 19500 | ✅ Seen-10：离散 20,000 + 连续 12,800<br>✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ Seen-10 离散/连续<br>✅ CrossMap-4 离散/连续 |
| `exp33` | 未开始，依赖 `exp31` | 未开始 | 未开始 |
| `exp33_loc` 100-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp33_loc` 50-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp33_loc` 20-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp33_loc` 10-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp33_gen` 100-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ CrossMap-4 离散/连续 |
| `exp33_gen` 50-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ CrossMap-4 离散/连续 |
| `exp33_gen` 20-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ CrossMap-4 离散/连续 |
| `exp33_gen` 10-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ CrossMap-4 离散/连续 |
| `exp34` | 未开始，依赖 `exp32` | 未开始 | 未开始 |
| `exp34_loc` 100-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp34_loc` 50-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp34_loc` 20-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp34_loc` 10-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp34_gen` 100-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ CrossMap-4 离散/连续 |
| `exp34_gen` 50-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ CrossMap-4 离散/连续 |
| `exp34_gen` 20-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ CrossMap-4 离散/连续 |
| `exp34_gen` 10-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ CrossMap-4 离散/连续 |
| `exp35_cs_office` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_gen_cs_office` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_loc_cs_office` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_de_golden` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_gen_de_golden` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_loc_de_golden` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_de_palacio` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_gen_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_loc_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_de_vertigo` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_gen_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_loc_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_cs_office` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp36_gen_cs_office` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_loc_cs_office` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_de_golden` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp36_gen_de_golden` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_loc_de_golden` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_de_palacio` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp36_gen_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_loc_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_de_vertigo` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp36_gen_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_loc_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp31_1` | 未开始 | 未开始 | 未开始 |
| `exp35_gen_cs_office` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_loc_cs_office` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_gen_de_golden` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous | 未开始 |
| `exp35_loc_de_golden` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_gen_de_palacio` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_palacio` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_gen_de_vertigo` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_vertigo` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_gen_cs_office` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_loc_cs_office` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_gen_de_golden` 50-shot | ✅ step 400 | 未开始 | 未开始 |
| `exp36_loc_de_golden` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_palacio` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_palacio` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_gen_de_vertigo` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_vertigo` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |



# csgo benchmark v2 主表

## 定位
| Setting | Task | Experiment | Shot/map | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Seen-10 | Localization | exp31 | - | 195.048 | 10.489 | 4.344 | 79.060 | 6000 |
| Seen-10 | Localization | exp31_loc | - | 79.294 | 4.134 | 2.708 | 36.173 | 19500 |
| Seen-10 | Localization | exp32 | - | 211.975 | 10.766 | 4.652 | 81.964 | 6000 |
| Seen-10 | Localization | exp32_loc | - | 48.961 | 3.006 | 2.970 | 26.047 | 19500 |
| CrossMap-4 few-shot | Localization | exp35_cs_office | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_golden | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_palacio | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_vertigo | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc_cs_office | 100 | 93.146 | 1.687 | 4.160 | 69.665 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_golden | 100 | 175.481 | 11.026 | 3.084 | 72.338 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_palacio | 100 | 218.156 | 12.130 | 3.304 | 65.144 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_vertigo | 100 | 164.352 | 16.383 | 3.916 | 72.533 | 400 |
| CrossMap-4 few-shot | Localization | exp36_cs_office | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_de_golden | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_de_palacio | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_de_vertigo | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_loc_cs_office | 100 | 78.451 | 1.657 | 4.261 | 65.293 | 400 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_golden | 100 | 141.977 | 10.249 | 2.537 | 70.545 | 400 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_palacio | 100 | 198.323 | 12.683 | 3.086 | 62.587 | 400 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_vertigo | 100 | 144.532 | 15.911 | 3.411 | 70.993 | 400 |
| CrossMap-4 few-shot | Localization | exp35 | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc | 100 | 162.784 | 10.307 | 3.616 | 69.920 | 400 |
| CrossMap-4 few-shot | Localization | exp36 | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_loc | 100 | 140.821 | 10.125 | 3.324 | 67.354 | 400 |
| Seen-10 retention | Localization | exp35_cs_office | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_golden | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_palacio | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_vertigo | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_loc_cs_office | 100 | 256.115 | 17.999 | 5.421 | 68.599 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_golden | 100 | 216.713 | 13.237 | 3.708 | 68.459 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_palacio | 100 | 217.588 | 12.125 | 4.046 | 59.172 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_vertigo | 100 | 208.859 | 13.430 | 4.537 | 59.888 | 400 |
| Seen-10 retention | Localization | exp36_cs_office | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp36_de_golden | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp36_de_palacio | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp36_de_vertigo | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp36_loc_cs_office | 100 | 231.142 | 22.456 | 4.182 | 36.080 | 400 |
| Seen-10 retention | Localization | exp36_loc_de_golden | 100 | 119.687 | 13.227 | 3.040 | 43.915 | 400 |
| Seen-10 retention | Localization | exp36_loc_de_palacio | 100 | 116.260 | 10.093 | 2.983 | 36.384 | 400 |
| Seen-10 retention | Localization | exp36_loc_de_vertigo | 100 | 145.808 | 7.935 | 3.287 | 34.537 | 400 |
| Seen-10 | Localization | exp31_1 | - |  |  |  |  | - |



## 离散生成
| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Discrete generation | exp31 | - | 13.212 | 0.4054 | 0.6425 | 0.4972 | 36.066 | 6000 |
| Seen-10 | Discrete generation | exp31_gen | - | 14.561 | 0.4268 | 0.5888 | 0.5258 | 28.629 | 19500 |
| Seen-10 | Discrete generation | exp32 | - | 12.995 | 0.3974 | 0.6583 | 0.4889 | 44.101 | 6000 |
| Seen-10 | Discrete generation | exp32_gen | - | 13.965 | 0.4113 | 0.6106 | 0.5163 | 35.584 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp31 | - |  |  |  |  |  | - |
| CrossMap-4 zero-shot | Discrete generation | exp31_gen | - | 12.288 | 0.4380 | 0.7358 | 0.4542 | 101.199 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp32 | - |  |  |  |  |  | - |
| CrossMap-4 zero-shot | Discrete generation | exp32_gen | - | 11.081 | 0.3690 | 0.7210 | 0.4592 | 79.887 | 19500 |
| CrossMap-4 few-shot | Discrete generation | exp35_cs_office | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_golden | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_palacio | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_vertigo | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 100 | 14.839 | 0.6121 | 0.5995 | 0.4686 | 34.197 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_golden | 100 | 15.506 | 0.3827 | 0.5924 | 0.5405 | 39.009 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_palacio | 100 | 13.630 | 0.3737 | 0.6682 | 0.5374 | 36.978 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_vertigo | 100 | 12.469 | 0.4603 | 0.6468 | 0.4609 | 39.447 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_cs_office | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_de_golden | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_de_palacio | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_de_vertigo | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_cs_office | 100 | 14.340 | 0.5913 | 0.6085 | 0.4673 | 34.787 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_golden | 100 | 15.160 | 0.3727 | 0.6012 | 0.5336 | 36.074 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_palacio | 100 | 13.205 | 0.3696 | 0.6743 | 0.5222 | 33.248 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_vertigo | 100 | 12.649 | 0.4548 | 0.6477 | 0.4578 | 35.587 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35 | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen | 100 | 14.111 | 0.4572 | 0.6267 | 0.5018 | 37.408 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36 | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_gen | 100 | 13.839 | 0.4471 | 0.6329 | 0.4952 | 34.924 | 400 |
| Seen-10 retention | Discrete generation | exp35_cs_office | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_golden | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_palacio | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_vertigo | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_gen_cs_office | 100 | 13.499 | 0.4629 | 0.6930 | 0.4798 | 116.424 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_golden | 100 | 13.449 | 0.3882 | 0.6298 | 0.5171 | 67.519 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_palacio | 100 | 13.377 | 0.4094 | 0.6583 | 0.4973 | 67.946 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_vertigo | 100 | 12.719 | 0.4287 | 0.6695 | 0.4746 | 75.466 | 400 |
| Seen-10 retention | Discrete generation | exp36_cs_office | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp36_de_golden | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp36_de_palacio | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp36_de_vertigo | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp36_gen_cs_office | 100 | 12.806 | 0.4319 | 0.6903 | 0.4698 | 94.478 | 400 |
| Seen-10 retention | Discrete generation | exp36_gen_de_golden | 100 | 13.070 | 0.3718 | 0.6554 | 0.5010 | 69.402 | 400 |
| Seen-10 retention | Discrete generation | exp36_gen_de_palacio | 100 | 12.724 | 0.3869 | 0.6746 | 0.4858 | 59.855 | 400 |
| Seen-10 retention | Discrete generation | exp36_gen_de_vertigo | 100 | 12.095 | 0.4035 | 0.6825 | 0.4676 | 78.333 | 400 |
| Seen-10 | Discrete generation | exp31_1 | - |  |  |  |  |  | - |
| CrossMap-4 zero-shot | Discrete generation | exp31_1 | - |  |  |  |  |  | - |



## 连续生成
| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | FVD↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Continuous generation | exp31 | - | 13.255 | 0.4026 | 0.6417 | 40.134 | 46.190 | 1011.744 | 6000 |
| Seen-10 | Continuous generation | exp31_gen | - | 15.164 | 0.4349 | 0.5622 | 31.409 | 38.033 | 737.197 | 19500 |
| Seen-10 | Continuous generation | exp32 | - | 13.078 | 0.3953 | 0.6601 | 40.511 | 46.401 | 1058.158 | 6000 |
| Seen-10 | Continuous generation | exp32_gen | - | 14.342 | 0.4162 | 0.5929 | 35.698 | 41.867 | 848.071 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp31 | - |  |  |  |  |  |  | - |
| CrossMap-4 zero-shot | Continuous generation | exp31_gen | - | 12.252 | 0.4291 | 0.7379 | 35.262 | 42.141 | 1131.397 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp32 | - |  |  |  |  |  |  | - |
| CrossMap-4 zero-shot | Continuous generation | exp32_gen | - | 11.042 | 0.3628 | 0.7208 | 46.932 | 52.377 | 1349.984 | 19500 |
| CrossMap-4 few-shot | Continuous generation | exp35_cs_office | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_golden | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_palacio | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_vertigo | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_cs_office | 100 | 14.849 | 0.5840 | 0.6113 | 20.140 | 28.299 | 455.915 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_golden | 100 | 15.598 | 0.3767 | 0.5798 | 19.058 | 25.349 | 706.815 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_palacio | 100 | 13.158 | 0.3490 | 0.6859 | 24.469 | 32.308 | 775.390 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_vertigo | 100 | 12.166 | 0.4447 | 0.6455 | 24.464 | 32.945 | 730.773 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_cs_office | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_de_golden | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_de_palacio | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_de_vertigo | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_cs_office | 100 | 14.284 | 0.5629 | 0.6254 | 26.457 | 33.674 | 570.938 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_golden | 100 | 15.211 | 0.3697 | 0.5928 | 25.736 | 31.647 | 870.651 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_palacio | 100 | 12.949 | 0.3514 | 0.6807 | 33.594 | 40.721 | 846.198 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_vertigo | 100 | 12.527 | 0.4449 | 0.6416 | 36.199 | 43.518 | 729.287 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35 | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen | 100 | 13.943 | 0.4386 | 0.6306 | 22.033 | 29.725 | 667.223 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36 | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_gen | 100 | 13.743 | 0.4322 | 0.6351 | 30.497 | 37.390 | 754.268 | 400 |
| Seen-10 retention | Continuous generation | exp35_cs_office | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_golden | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_palacio | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_vertigo | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_gen_cs_office | 100 | 13.619 | 0.4645 | 0.6864 | 25.670 | 33.345 | 1655.739 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_golden | 100 | 13.546 | 0.3867 | 0.6287 | 24.207 | 31.901 | 909.982 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_palacio | 100 | 13.783 | 0.4174 | 0.6501 | 26.768 | 34.261 | 993.103 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_vertigo | 100 | 13.070 | 0.4332 | 0.6642 | 31.365 | 38.412 | 1143.649 | 400 |
| Seen-10 retention | Continuous generation | exp36_cs_office | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp36_de_golden | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp36_de_palacio | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp36_de_vertigo | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp36_gen_cs_office | 100 | 12.830 | 0.4315 | 0.6910 | 33.353 | 39.939 | 1518.554 | 400 |
| Seen-10 retention | Continuous generation | exp36_gen_de_golden | 100 | 13.114 | 0.3697 | 0.6590 | 29.135 | 36.264 | 1004.852 | 400 |
| Seen-10 retention | Continuous generation | exp36_gen_de_palacio | 100 | 12.951 | 0.3915 | 0.6693 | 34.706 | 41.255 | 1011.143 | 400 |
| Seen-10 retention | Continuous generation | exp36_gen_de_vertigo | 100 | 12.217 | 0.4036 | 0.6819 | 40.245 | 46.283 | 1258.807 | 400 |
| Seen-10 | Continuous generation | exp31_1 | - |  |  |  |  |  |  | - |
| CrossMap-4 zero-shot | Continuous generation | exp31_1 | - |  |  |  |  |  |  | - |



# csgo benchmark v2 补充表格

## 定位
| Setting | Task | Experiment | Shot/map | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp33 | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp33_loc | 100 | 175.491 | 11.717 | 3.968 | 76.271 |
| CrossMap-4 few-shot | Localization | exp33_loc | 50 | 195.684 | 13.497 | 3.995 | 84.102 |
| CrossMap-4 few-shot | Localization | exp33_loc | 20 | 199.455 | 14.185 | 3.662 | 82.913 |
| CrossMap-4 few-shot | Localization | exp33_loc | 10 | 216.629 | 14.493 | 3.790 | 86.198 |
| CrossMap-4 few-shot | Localization | exp34 | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp34_loc | 100 | 155.607 | 11.146 | 3.516 | 73.515 |
| CrossMap-4 few-shot | Localization | exp34_loc | 50 | 187.740 | 14.105 | 3.992 | 80.578 |
| CrossMap-4 few-shot | Localization | exp34_loc | 20 | 193.382 | 13.611 | 4.235 | 82.967 |
| CrossMap-4 few-shot | Localization | exp34_loc | 10 | 214.016 | 13.774 | 4.187 | 86.504 |
| Seen-10 retention | Localization | exp33 | 100 |  |  |  |  |
| Seen-10 retention | Localization | exp33_loc | 100 | 232.186 | 19.489 | 3.852 | 70.011 |
| Seen-10 retention | Localization | exp33_loc | 50 | 209.635 | 12.769 | 3.772 | 63.909 |
| Seen-10 retention | Localization | exp33_loc | 20 | 213.673 | 12.539 | 3.745 | 64.786 |
| Seen-10 retention | Localization | exp33_loc | 10 | 210.105 | 15.243 | 3.529 | 65.665 |
| Seen-10 retention | Localization | exp34 | 100 |  |  |  |  |
| Seen-10 retention | Localization | exp34_loc | 100 | 132.404 | 10.013 | 3.046 | 34.489 |
| Seen-10 retention | Localization | exp34_loc | 50 | 112.538 | 9.760 | 3.037 | 37.961 |
| Seen-10 retention | Localization | exp34_loc | 20 | 133.374 | 14.983 | 3.606 | 40.931 |
| Seen-10 retention | Localization | exp34_loc | 10 | 137.350 | 18.550 | 3.139 | 48.072 |
| CrossMap-4 few-shot | Localization | exp35_loc_cs_office | 50 | 121.609 | 2.313 | 4.390 | 84.044 |
| Seen-10 retention | Localization | exp35_loc_cs_office | 50 | 267.171 | 17.694 | 7.181 | 68.238 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_golden | 50 | 182.041 | 13.299 | 3.023 | 82.515 |
| Seen-10 retention | Localization | exp35_loc_de_golden | 50 | 232.139 | 18.689 | 3.783 | 72.837 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_palacio | 50 | 239.941 | 13.969 | 3.574 | 72.311 |
| Seen-10 retention | Localization | exp35_loc_de_palacio | 50 | 228.142 | 16.225 | 3.717 | 63.418 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_vertigo | 50 | 166.369 | 16.331 | 3.869 | 76.606 |
| Seen-10 retention | Localization | exp35_loc_de_vertigo | 50 | 221.817 | 13.054 | 3.903 | 63.401 |
| CrossMap-4 few-shot | Localization | exp36_loc_cs_office | 50 | 108.359 | 2.364 | 4.890 | 78.328 |
| Seen-10 retention | Localization | exp36_loc_cs_office | 50 | 227.768 | 27.347 | 4.594 | 42.501 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_golden | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_golden | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_palacio | 50 | 236.262 | 16.242 | 3.250 | 74.013 |
| Seen-10 retention | Localization | exp36_loc_de_palacio | 50 | 182.345 | 14.171 | 3.322 | 33.873 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_vertigo | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_vertigo | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc | 50 | 177.490 | 11.478 | 3.714 | 78.869 |
| CrossMap-4 few-shot | Localization | exp36_loc | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc_cs_office | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp35_loc_cs_office | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc_de_golden | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp35_loc_de_golden | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc_de_palacio | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp35_loc_de_palacio | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc_de_vertigo | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp35_loc_de_vertigo | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_cs_office | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_cs_office | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_golden | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_golden | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_palacio | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_palacio | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_vertigo | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_vertigo | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc_cs_office | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp35_loc_cs_office | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc_de_golden | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp35_loc_de_golden | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc_de_palacio | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp35_loc_de_palacio | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc_de_vertigo | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp35_loc_de_vertigo | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_cs_office | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_cs_office | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_golden | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_golden | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_palacio | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_palacio | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_vertigo | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_vertigo | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp35_loc | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc | 10 |  |  |  |  |



## 离散生成
| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp33_gen | 100 | 13.520 | 0.4413 | 0.6475 | 0.4867 | 35.407 |
| CrossMap-4 few-shot | Discrete generation | exp33_gen | 50 | 13.462 | 0.4623 | 0.6590 | 0.4785 | 48.555 |
| CrossMap-4 few-shot | Discrete generation | exp33_gen | 20 | 13.267 | 0.4509 | 0.6654 | 0.4781 | 53.425 |
| CrossMap-4 few-shot | Discrete generation | exp33_gen | 10 | 12.795 | 0.4483 | 0.6815 | 0.4676 | 67.969 |
| CrossMap-4 few-shot | Discrete generation | exp34_gen | 100 | 13.138 | 0.4309 | 0.6613 | 0.4826 | 38.850 |
| CrossMap-4 few-shot | Discrete generation | exp34_gen | 50 | 13.264 | 0.4419 | 0.6599 | 0.4771 | 43.035 |
| CrossMap-4 few-shot | Discrete generation | exp34_gen | 20 | 13.096 | 0.4362 | 0.6652 | 0.4723 | 44.739 |
| CrossMap-4 few-shot | Discrete generation | exp34_gen | 10 | 12.784 | 0.4283 | 0.6756 | 0.4642 | 52.337 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 50 | 13.952 | 0.6020 | 0.6342 | 0.4448 | 45.005 |
| Seen-10 retention | Discrete generation | exp35_gen_cs_office | 50 | 13.303 | 0.4660 | 0.7066 | 0.4739 | 129.509 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_golden | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_golden | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_palacio | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_palacio | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_vertigo | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_vertigo | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_cs_office | 50 | 13.584 | 0.5890 | 0.6393 | 0.4501 | 38.422 |
| Seen-10 retention | Discrete generation | exp36_gen_cs_office | 50 | 12.823 | 0.4382 | 0.6977 | 0.4656 | 103.438 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_golden | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_golden | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_palacio | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_palacio | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_vertigo | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_vertigo | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_cs_office | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_golden | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_golden | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_palacio | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_palacio | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_vertigo | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_vertigo | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_cs_office | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_cs_office | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_golden | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_golden | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_palacio | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_palacio | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_vertigo | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_vertigo | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_cs_office | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_golden | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_golden | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_palacio | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_palacio | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_vertigo | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp35_gen_de_vertigo | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_cs_office | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_cs_office | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_golden | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_golden | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_palacio | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_palacio | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_vertigo | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_vertigo | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp35_gen | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen | 10 |  |  |  |  |  |



## 连续生成
| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | FVD↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp33_gen | 100 | 13.233 | 0.4230 | 0.6555 | 31.991 | 38.899 | 853.537 |
| CrossMap-4 few-shot | Continuous generation | exp33_gen | 50 | 13.061 | 0.4458 | 0.6731 | 22.406 | 30.492 | 839.893 |
| CrossMap-4 few-shot | Continuous generation | exp33_gen | 20 | 12.774 | 0.4315 | 0.6763 | 21.927 | 29.860 | 896.970 |
| CrossMap-4 few-shot | Continuous generation | exp33_gen | 10 | 12.532 | 0.4301 | 0.6842 | 22.488 | 30.143 | 955.571 |
| CrossMap-4 few-shot | Continuous generation | exp34_gen | 100 | 12.896 | 0.4125 | 0.6691 | 38.268 | 44.353 | 934.424 |
| CrossMap-4 few-shot | Continuous generation | exp34_gen | 50 | 12.935 | 0.4236 | 0.6671 | 31.576 | 38.439 | 975.876 |
| CrossMap-4 few-shot | Continuous generation | exp34_gen | 20 | 12.805 | 0.4182 | 0.6700 | 31.074 | 37.991 | 895.785 |
| CrossMap-4 few-shot | Continuous generation | exp34_gen | 10 | 12.521 | 0.4106 | 0.6796 | 31.051 | 37.968 | 911.294 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_cs_office | 50 | 13.214 | 0.5588 | 0.6678 | 19.668 | 27.716 | 679.993 |
| Seen-10 retention | Continuous generation | exp35_gen_cs_office | 50 | 13.407 | 0.4670 | 0.7045 | 26.690 | 34.130 | 1699.999 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_golden | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_golden | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_palacio | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_palacio | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_vertigo | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_vertigo | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_cs_office | 50 | 12.967 | 0.5558 | 0.6730 | 29.120 | 36.126 | 610.194 |
| Seen-10 retention | Continuous generation | exp36_gen_cs_office | 50 | 12.789 | 0.4353 | 0.6978 | 33.573 | 40.052 | 1612.398 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_golden | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_golden | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_palacio | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_palacio | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_vertigo | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_vertigo | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_cs_office | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_cs_office | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_golden | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_golden | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_palacio | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_palacio | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_vertigo | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_vertigo | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_cs_office | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_cs_office | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_golden | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_golden | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_palacio | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_palacio | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_vertigo | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_vertigo | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_cs_office | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_cs_office | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_golden | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_golden | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_palacio | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_palacio | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_vertigo | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp35_gen_de_vertigo | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_cs_office | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_cs_office | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_golden | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_golden | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_palacio | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_palacio | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_vertigo | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_vertigo | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp35_gen | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen | 10 |  |  |  |  |  |  |
