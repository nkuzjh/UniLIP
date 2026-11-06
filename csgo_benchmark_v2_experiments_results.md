# csgo benchmark v2 实验设计与表格维护规则
===========================================

最新 Seen-3 LoRA 补充（2026-11-01）：U4／B1 与独立 B2 的 local1200／1800 validation、local1800 test 已登记；共 21 个完成模型，其中 18 个 Stage 2。原九组 9/9，计入 P5／U5 的 P／U 系列 10/10。[U4／B1 指标补充](#u4-b1-results-20261101)与[B2 消融补充](#exp32-3maps-b2-s3-j8-20261030)见下方；旧日期统计保留当时口径。

## 1. exp31~exp36 实验设计定位

| 实验系列 | 设计定位 |
|---|---|
| `exp31` | Seen-10 full-head 实验；`exp31` 为 joint 联合训练且开启 aux_loc_loss 与 perception_loss，`exp31_1` 为关闭 aux 与 perception 的 loss-only control，`exp31_gen`/`exp31_loc` 为单任务实验，评测  Seen-10 与CrossMap zero-shot。 |
| `exp31*_3maps` | Benchmark v2 Seen-3 full-head 联合训练低成本 `2 x 2` loss ablation；固定使用 `de_ancient`、`de_dust2`、`de_nuke`，分别覆盖 aux+perception、joint-only、aux-only、perception-only，只评测 Seen-3。 |
| `exp32_3maps_*` | Benchmark v2 Seen-3 LoRA 消融，共 24 个独立运行名；固定同三图，原两轮的第一轮 L0/G0/J0～J4 共 7 组、补充联合对照 J5～J10 共 6 组比较单任务、联合训练、学习率和 αloc，第二轮原 A0～A3 共 4 组（J4）及 2026-10-14 新增 A1_J8～A3_J8 共 3 组（J8）比较从原始权重训练的辅助 loss；另有从 J8@5850 权重开始的独立 Stage 2 四组 S0_J8～S3_J8（无辅助对照＋三个辅助主组）。使用 seen_validation 分阶段筛选，选定 checkpoint 后报告 Seen-3 离散／连续测试。 |
| `exp32` | Seen-10 LoRA 实验；`exp32` 为 joint 联合训练且关闭 aux_loc_loss 与 perception_loss，`exp32_gen`/`exp32_loc` 为单任务实验，评测  Seen-10 与CrossMap zero-shot。 |
| `exp33`/`exp34` | 分别从 matching 的 `exp31*`/`exp32*` 初始化，在 CrossMap-4 四图共同 support 上进行非 map-specific 的 `100/50/20/10-shot` 适配；包含 joint、gen、loc 路由，并评测 CrossMap 与 Seen-10 retention。 |
| `exp35`/`exp36` | 分别从 matching 的 `exp31*`/`exp32*` 初始化，对 `cs_office`、`de_golden`、`de_palacio`、`de_vertigo` 四张 CrossMap 地图分别进行 map-specific 适配；shot 为 `100/50/20/10`，包含 joint、gen、loc 路由，评测 CrossMap 与 Seen-10 retention。 |
| `exp36_1` | 从 `exp31_1` 初始化的 map-specific full-head joint-only 对照；不存在 gen/loc 子训练路由，评测 CrossMap 与 Seen-10 retention。 |

结果表路由规则如下：joint 结果进入定位、离散生成、连续生成三个表；gen-only 结果进入两张生成表；loc-only 结果只进入定位表。map-specific 实验记录每个地图 checkpoint 的 CrossMap 结果和 Seen-10 retention；无 map 后缀的 family 名称只表示四个目标地图 CrossMap 结果的等地图宏平均，不是额外训练实验，也不要求对四个 retention checkpoint 再做 family 平均。非 map-specific 的 `exp33`/`exp34` 使用一个 checkpoint 同时覆盖 CrossMap-4，结果按 shot 记录跨地图聚合。

## 2. 实验归属与表格对应

- **ablation 3 maps实验进度表及ablation 3 maps表**：`exp31_3maps`、`exp31_1_3maps`、`exp31_2_3maps`、`exp31_3_3maps`，以及 `exp32_3maps_*` 原第一轮 7 组、补充联合对照 J5～J10 共 6 组、原第二轮 A0～A3 共 4 组、A1_J8～A3_J8 共 3 组及独立 Stage 2 S0_J8～S3_J8 共 4 组，合计 24 个独立运行名。
- **主线进度表及主表**：`exp31*`、`exp32*`、`exp35*`、`exp36_1*`。
- **暂停支线进度表及补充表格**：`exp33*`、`exp34*`、`exp36*`。

上述归属是语义映射，不按实验名称字符串精确匹配。`exp31*_3maps` 和 `exp32_3maps_*` 必须进入独立的 ablation 表，不进入 Seen-10 主表。派生的宏平均、retention 结果以及按任务拆分的结果表行，都属于对应的同一个进度实验。

## 3. 排序保护

- 当前顺序是用户手工确认的 canonical order（规范顺序）。
- 更新实验结果时只能原位填写对应单元格，严格禁止全局排序、按名称排序、重建表、移动已有行、把空结果移到末尾或改变分隔行。
- ablation、主表和补充表格中的定位、离散生成、连续生成三个任务表，其现有相对顺序均不可改变。
- `ablation 3 maps表` 必须保留当前两行 Setext 一级标题格式（标题下一行 `====================`），以兼容已启动的旧版 map-specific runner；禁止改写为 `#` 标题。

## 4. 新实验加入流程

新增实验时，先确定 parent、训练路线（joint/gen/loc）、Seen 或 CrossMap、完整协议或固定地图子集、zero-shot 或 few-shot、是否 map-specific、地图、shot、评测任务、retention 和 macro；再判断其属于 ablation、主线还是暂停支线。随后按照当前最近的同族语义顺序，将实验加入对应的进度表和适用的任务结果表。map-specific 实验同时预留逐图结果和 CrossMap family macro；只添加必要的任务表，不制造不存在的 gen/loc 子实验。插入后核对三个任务表的语义投影、地图顺序、shot 顺序、retention 和宏平均。



# csgo benchmark v2 实验进度

## ablation 3 maps实验进度表

| 实验版本 | 训练 | 推理 | Metric 计算 | validation | checkpoint steps |
| --- | --- | --- | --- | --- | --- |
| `exp31_3maps` | 未开始；目标 50 epochs，预计 step 5900 | 未开始；Seen-3 定位 6,000 + 离散生成 6,000 + 连续生成 3,840 | 未开始 | — | — |
| `exp31_1_3maps` | ✅ Seen-3，50 epochs，step 5850 | ✅ Seen-3：定位 6,000 + 离散生成 6,000 + 连续生成 3,840 | ✅ Seen-3：定位/离散/连续 metric 与三地图聚合完成 | — | — |
| `exp31_2_3maps` | ✅ Seen-3，50 epochs， step 5900 | ✅ Seen-3 定位 6,000 + 离散生成 6,000 + 连续生成 3,840 | ✅ Seen-3：定位/离散/连续 metric 与三地图聚合完成 | — | — |
| `exp31_3_3maps` | ✅ Seen-3，50 epochs，step 5900 | ✅ Seen-3：定位 6,000 + 离散生成 6,000 + 连续生成 3,840 | ✅ Seen-3：定位/离散/连续 metric 与三地图聚合完成 | — | — |
| `exp32_3maps_loc_single`（L0） | ✅ 本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位 2400／3600／5850；test 待评测 | ✅ seen_validation 定位 三图指标；test 待评测 | ✅ seen_validation 定位，三图各 500 条；seed=42 | 本地 checkpoint：2400、3600、5850；已评测 2400、3600、5850 |
| `exp32_3maps_gen_single`（G0） | ✅ 本地核实 50 epochs，5850/5850 | ✅ seen_validation 离散生成 2400／3600／5850；test 待评测 | ✅ seen_validation 离散生成 三图指标；test 待评测 | ✅ seen_validation 离散生成，三图各 500 条；seed=42 | 本地 checkpoint：2400、3600、5850；已评测 2400、3600、5850 |
| `exp32_3maps_joint_original`（J0） | ✅ 本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成 三图指标；test 待评测 | ✅ seen_validation 定位＋离散生成，三图各 500 条；seed=42 | 本地 checkpoint：2400、3600、5850；已评测 2400、5850 |
| `exp32_3maps_joint_headlr`（J1） | ✅ 本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成 三图指标；test 待评测 | ✅ seen_validation 定位＋离散生成，三图各 500 条；seed=42 | 本地 checkpoint：2400、3600、5850；已评测 2400、5850 |
| `exp32_3maps_joint_sharedlr`（J2） | ✅ 本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成 三图指标；test 待评测 | ✅ seen_validation 定位＋离散生成，三图各 500 条；seed=42 | 本地 checkpoint：2400、3600、5850；已评测 2400、5850 |
| `exp32_3maps_joint_constloc`（J3） | ✅ 本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成 三图指标；test 待评测 | ✅ seen_validation 定位＋离散生成，三图各 500 条；seed=42 | 本地 checkpoint：2400、3600、5850；已评测 2400、5850 |
| `exp32_3maps_joint_sharedlr_constloc`（J4） | ✅ 本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成 三图指标；test 待评测 | ✅ seen_validation 定位＋离散生成，三图各 500 条；seed=42 | 本地 checkpoint：2400、3600、4800、5850；已评测 2400、5850 |
| `exp32_3maps_joint_original_constloc`（J5） | ✅ 远端完成；本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成三图指标；test 待评测 | ✅ seen_validation 两任务各三图各 500 条；seed=42 | 本地模型：1200、2400、3600、4800、5850 及根目录；已评测 2400、5850 |
| `exp32_3maps_joint_constloc1`（J6） | ✅ 远端完成；本地状态核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成三图指标；test 待评测 | ✅ seen_validation 两任务各三图各 500 条；seed=42 | 本地 2400、5850 模型文件已到；其余同步中 |
| `exp32_3maps_joint_constloc4`（J7） | ✅ 远端完成；本地状态核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成三图指标；test 待评测 | ✅ seen_validation 两任务各三图各 500 条；seed=42 | 本地 2400、5850 模型文件已到；其余同步中 |
| `exp32_3maps_joint_constloc8`（J8） | ✅ 远端完成；本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成三图指标；test 待评测 | ✅ seen_validation 两任务各三图各 500 条；seed=42 | 本地模型：2400、5850 及根目录；其余同步中 |
| `exp32_3maps_joint_constloc12`（J9） | ✅ 远端完成；本地核实 50 epochs，5850/5850 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成三图指标；test 待评测 | ✅ seen_validation 两任务各三图各 500 条；seed=42 | 本地模型：1200、2400、3600、4800、5850 及根目录；已评测 2400、5850 |
| `exp32_3maps_joint_constloc16`（J10） | 远端完成（用户确认）；本地训练元数据待同步 | ✅ seen_validation 定位＋离散生成 2400／5850；test 待评测 | ✅ seen_validation 定位＋离散生成三图指标；test 待评测 | ✅ seen_validation 两任务各三图各 500 条；seed=42 | 本地模型与元数据待同步；已评测 2400、5850 |
| `exp32_3maps_aux_control`（A0） | 待记录；首阶段目标 step 3600；总预算 50 epochs | 待记录；定位＋离散 validation／test，连续仅 test | 待记录；定位推理内计算，生成 metric 单独执行 | 待记录；seen_validation 定位 1,500＋离散生成 1,500；seed=42 | 计划：3600（首评）；4800／实际最终 step（入围后） |
| `exp32_3maps_perception`（A1） | 待记录；首阶段目标 step 3600；总预算 50 epochs | 待记录；定位＋离散 validation／test，连续仅 test | 待记录；定位推理内计算，生成 metric 单独执行 | 待记录；seen_validation 定位 1,500＋离散生成 1,500；seed=42 | 计划：3600（首评）；4800／实际最终 step（入围后） |
| `exp32_3maps_auxloc`（A2） | 待记录；首阶段目标 step 3600；总预算 50 epochs | 待记录；定位＋离散 validation／test，连续仅 test | 待记录；定位推理内计算，生成 metric 单独执行 | 待记录；seen_validation 定位 1,500＋离散生成 1,500；seed=42 | 计划：3600（首评）；4800／实际最终 step（入围后） |
| `exp32_3maps_both`（A3） | 仅发现 J4 配方计划／配置；无训练完成证据；首阶段目标 step 3600 | 待记录；定位＋离散 validation／test，连续仅 test | 待记录；定位推理内计算，生成 metric 单独执行 | 待记录；seen_validation 定位 1,500＋离散生成 1,500；seed=42 | 计划：3600（首评）；4800／实际最终 step（入围后） |
| `exp32_3maps_perception_j8`（A1_J8） | 已配置／未启动；J8 固定配方，首停 3600，50 epochs horizon | 待评测；定位＋离散 validation，选定后 test | 待评测；定位推理内计算，生成 metric 单独执行 | 计划 seen_validation 两任务各 1,500 条；seed=42 | 计划：3600（首评）；入围后续训至匹配步数 |
| `exp32_3maps_auxloc_j8`（A2_J8） | 已配置／未启动；J8 固定配方，首停 3600，50 epochs horizon | 待评测；定位＋离散 validation，选定后 test | 待评测；定位推理内计算，生成 metric 单独执行 | 计划 seen_validation 两任务各 1,500 条；seed=42 | 计划：3600（首评）；入围后续训至匹配步数 |
| `exp32_3maps_both_j8`（A3_J8） | 已配置／未启动；J8 固定配方，首停 3600，50 epochs horizon | 待评测；定位＋离散 validation，选定后 test | 待评测；定位推理内计算，生成 metric 单独执行 | 计划 seen_validation 两任务各 1,500 条；seed=42 | 计划：3600（首评）；入围后续训至匹配步数 |

| `exp32_3maps_stage2_control_j8`（S0_J8） | 已配置／未启动；J8@5850 权重初始化，Stage 2 1800 local steps、首停 1200 | 待评测；定位＋离散 validation，选定后 test | 待评测；定位推理内计算，生成 metric 单独执行 | 计划 seen_validation 两任务各三图各 500 条；seed=42 | 计划 local 600／1200；按需 1800；尚无 Stage 2 checkpoint |
| `exp32_3maps_stage2_perception_j8`（S1_J8） | 已配置／未启动；J8@5850 权重初始化，Stage 2 1800 local steps、首停 1200 | 待评测；定位＋离散 validation，选定后 test | 待评测；定位推理内计算，生成 metric 单独执行 | 计划 seen_validation 两任务各三图各 500 条；seed=42 | 计划 local 600／1200；按需 1800；尚无 Stage 2 checkpoint |
| `exp32_3maps_stage2_auxloc_j8`（S2_J8） | 已配置／未启动；J8@5850 权重初始化，Stage 2 1800 local steps、首停 1200 | 待评测；定位＋离散 validation，选定后 test | 待评测；定位推理内计算，生成 metric 单独执行 | 计划 seen_validation 两任务各三图各 500 条；seed=42 | 计划 local 600／1200；按需 1800；尚无 Stage 2 checkpoint |
| `exp32_3maps_stage2_both_j8`（S3_J8） | 已配置／未启动；J8@5850 权重初始化，Stage 2 1800 local steps、首停 1200 | 待评测；定位＋离散 validation，选定后 test | 待评测；定位推理内计算，生成 metric 单独执行 | 计划 seen_validation 两任务各三图各 500 条；seed=42 | 计划 local 600／1200；按需 1800；尚无 Stage 2 checkpoint |
| `exp32_3maps_B2_S3_J8`（B2；2026-10-30 新增；2026-11-01 同步） | ✅ J8@5850 仅模型初始化，local1800/1800；双卡 micro4/GAS16；[状态](outputs/csgo_1b/exp32_3maps_B2_S3_J8/trainer_state.json) | ✅ validation1200／1800 定位＋离散；test1800 定位＋离散＋连续 | ✅ 对应三项宏平均与逐图指标齐全 | ✅ seen_validation local1200／1800 三图各500，seed=42 | 当前本地模型 local1200／1800；[计划](outputs/csgo_1b/exp32_3maps_B2_S3_J8/experiment_plan.json)、[配方](outputs/csgo_1b/exp32_3maps_B2_S3_J8/resolved_config.yaml)；未核实完整恢复材料 |

上述 S0_J8～S3_J8 四行是 2026-10-17 配置登记时的计划快照；当前结果及 2026-11-01 新增 U4／B1／B2 同步状态见[主实验文档](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#test-overall-20261101)与下方补充。

### S0_G0／S0_L0 续训进度（2026-10-31 同步）

| 实验版本 | 训练 | 推理 | Metric 计算 | validation | checkpoint steps |
|---|---|---|---|---|---|
| `exp32_3maps_stage2_control_l0`（S0_L0；2026-10-30 同步） | ✅ L0@5850 初始化；local1800/1800，epoch=15.256；[Trainer 状态](outputs/csgo_1b/exp32_3maps_stage2_control_l0/trainer_state.json) | ✅ 定位 validation1200／1800 各1,500条；test1800 共6,000条；生成 N/A | ✅ 三节点定位宏平均与逐图指标齐全 | 已完成 local1200／1800，seen_validation 三图各500；seed=42 | 每600步保存；本地现有 local1800＋根目录最终模型；未见600／1200模型，1200指标已同步；test固定local1800 |
| `exp32_3maps_stage2_control_g0`（S0_G0；2026-10-31 同步） | ✅ G0@5850 初始化；local1800/1800，epoch=15.256；[Trainer 状态](outputs/csgo_1b/exp32_3maps_stage2_control_g0/trainer_state.json) | ✅ 离散 validation1200／1800 各1,500条；test1800 离散6,000条＋连续3,840帧；定位 N/A | ✅ 四份生成评测 summary 与逐图指标齐全 | 已完成 local1200／1800，seen_validation 三图各500；seed=42 | 每600步保存；本地现有 checkpoint-600／1200／1800 与根目录最终模型；test固定local1800 |

上述 exp32_3maps 第一轮结果的源 JSON 与运行证据见 [两轮实验记录和结果表](CSGO_BENCHMARK_V2_EXP32_3MAPS.md)。J5～J10 的远端训练完成由用户确认；截至 2026-10-13 的同步快照，J5～J9 的本地 Trainer 状态均为 50 epochs、5850/5850、train batch=64，运行计划均记录单 GPU、microbatch=64、GAS=2、seed=42。J5 本地模型 1200／2400／3600／4800／5850 均已到；J6/J7/J8 本地模型 2400／5850 已到；J9 本地模型 1200／2400／3600／4800／5850 已到；上述各组均有根目录模型。J10 的本地训练元数据及模型尚待同步。J5～J9 所列模型仅核对 safetensors 头部声明的长度与文件大小吻合，未做校验和或 optimizer、scheduler、RNG 等完整恢复文件核查；快照随同步变化。其余计划节点不代表已生成 checkpoint；不依据命令示例推断实际运行进度。`validation` 列记录筛选集及验证状态，`checkpoint steps` 列按各行说明区分本地已有与计划评测节点；旧 exp31 行的两列暂记为 `—`，原有进度保留。原两轮实验每 1200 optimizer steps 保存完整 checkpoint，1200 为可选异常检查点；第一轮默认在 2400 停止，第二轮默认在 3600 停止，随后手动推理、计算指标及决定是否续训。最终 step 以实际训练为准，约 5900 仅为估计。设置与命令见 [record.md](record.md) 和 [两轮实验说明](CSGO_BENCHMARK_V2_EXP32_3MAPS.md)。


2026-10-14 新配置仅登记 A1_J8～A3_J8 的计划状态，以上 2026-10-13 训练及评测证据快照日期保持不变。原 A0～A3 名称、J4 默认配方及已存在输出不覆盖。J8 为新组三者的无辅助参照，仅在原始初始化、seed、主配方、数据布局、有效 batch 和评测步数一致时使用；本地 J8 现有结果仅为 2400／5850，3600 对照需补评 J8@3600。


2026-10-17 新增独立 J8 Stage 2 四组 S0_J8～S3_J8，累计 24 个 exp32_3maps 运行名。S1～S3 为三个辅助主组；S0 是匹配更新数、主任务曝光与 LR 的无辅助归因对照，辅助计算的 GPU 时间不匹配。四组均从 J8@5850 模型权重开始，Stage 2 使用新 optimizer／local step 0／1800 步调度；旧 A1_J8～A3_J8 仍为从原始权重独立训练的 50 epoch 配方。四组尚未启动，结果待填；既有 J8@5850 可作初始化参照，辅助效应以 S0 同 local checkpoint 比较，报告两阶段总成本时计入共享的 J8 Stage 1 成本。详见 [Stage 2 设计](CSGO_BENCHMARK_V2_EXP32_3MAPS_STAGE2.md)及 [执行命令](record.md)。

## 主线实验进度表

| 实验版本 | 训练 | 推理 | Metric 计算 |
|---|---|---|---|
| `exp31` | ✅ Seen-10，step 19550；✅ 评测 checkpoint-6000 | ✅ Seen-10：定位 20,000 + 离散生成 20,000 + 连续生成 12,800<br>✅ CrossMap-4 zero-shot | ✅ Seen-10：定位/离散/连续 metric 与聚合完成<br>✅ CrossMap-4 zero-shot |
| `exp31_1` | ✅ Seen-10，step 19500 | ✅ Seen-10：定位 20,000 + 离散生成 20,000 + 连续生成 12,800<br>✅ CrossMap-4 zero-shot：定位 8,000 + 离散生成 8,000 + 连续生成 5,120 | ✅ Seen-10 与 CrossMap-4 zero-shot：定位/离散/连续 metric 与聚合完成 |
| `exp31_gen` | ✅ Seen-10，step 19500 | ✅ Seen-10：离散 20,000 + 连续 12,800<br>✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ Seen-10 离散/连续<br>✅ CrossMap-4 离散/连续 |
| `exp31_loc` | ✅ Seen-10，step 19500 | ✅ Seen-10，20,000 样本<br>CrossMap-4 zero-shot：未开始 | ✅ Seen-10 定位 metric<br>CrossMap-4 zero-shot：未开始 |
|---|---|---|---|
| `exp32` | ✅ Seen-10，step 19550；✅  checkpoint-6000 | ✅ checkpoint-19550 Seen-10 ：定位 20,000 + 离散生成 20,000 + 连续生成 12,800<br>CrossMap-4 zero-shot；checkpoint-6000 未开始| ✅ checkpoint-19550 Seen-10 ：定位/离散/连续 metric 与聚合完成<br>CrossMap-4 zero-shot；checkpoint-6000未开始 |
| `exp32_gen` | ✅ Seen-10，step 19500 | ✅ Seen-10：离散 20,000 + 连续 12,800<br>✅ CrossMap-4：离散 8,000 + 连续 5,120 | ✅ Seen-10 离散/连续<br>✅ CrossMap-4 离散/连续 |
| `exp32_loc` | ✅ Seen-10，step 19500 | ✅ Seen-10，20,000 样本<br>CrossMap-4 zero-shot：未开始 | ✅ Seen-10 定位 metric<br>CrossMap-4 zero-shot：未开始 |
|---|---|---|---|
| `exp35_cs_office` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_golden` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_palacio` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_vertigo` 100-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp36_1_cs_office` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_golden` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp35_gen_cs_office` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_gen_de_golden` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_gen_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_gen_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_loc_cs_office` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_loc_de_golden` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_loc_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_loc_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
|---|---|---|---|
| `exp35_cs_office` 50-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_golden` 50-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_palacio` 50-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_vertigo` 50-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp36_1_cs_office` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_golden` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_palacio` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_vertigo` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp35_gen_cs_office` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_gen_de_golden` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_gen_de_palacio` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp35_gen_de_vertigo` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | 未开始 |
| `exp35_loc_cs_office` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_loc_de_golden` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_loc_de_palacio` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp35_loc_de_vertigo` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
|---|---|---|---|
| `exp35_cs_office` 20-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_golden` 20-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_palacio` 20-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_vertigo` 20-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp36_1_cs_office` 20-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_golden` 20-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_palacio` 20-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_vertigo` 20-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp35_gen_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
|---|---|---|---|
| `exp35_cs_office` 10-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_golden` 10-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_palacio` 10-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp35_de_vertigo` 10-shot | 等待 parent checkpoint | 未开始 | 未开始 |
| `exp36_1_cs_office` 10-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_golden` 10-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_palacio` 10-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp36_1_de_vertigo` 10-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous<br>CrossMap-4 localization<br>Seen-10 retention localization | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous<br>CrossMap localization<br>Seen retention localization |
| `exp35_gen_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_gen_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp35_loc_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |
|---|---|---|---|



## 暂停的支线实验进度表

| 实验版本 | 训练 | 推理 | Metric 计算 |
|---|---|---|---|
| `exp33` | 未开始，依赖 `exp31` | 未开始 | 未开始 |
| `exp33_gen` 100-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120<br>Seen-10 retention：未开始 | ✅ CrossMap-4 离散/连续<br>Seen-10 retention：未开始 |
| `exp33_gen` 50-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120<br>Seen-10 retention：未开始 | ✅ CrossMap-4 离散/连续<br>Seen-10 retention：未开始 |
| `exp33_gen` 20-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120<br>Seen-10 retention：未开始 | ✅ CrossMap-4 离散/连续<br>Seen-10 retention：未开始 |
| `exp33_gen` 10-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120<br>Seen-10 retention：未开始 | ✅ CrossMap-4 离散/连续<br>Seen-10 retention：未开始 |
| `exp33_loc` 100-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp33_loc` 50-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp33_loc` 20-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp33_loc` 10-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
|---|---|---|---|
| `exp34` | 未开始 | 未开始 | 未开始 |
| `exp34_gen` 100-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120<br>Seen-10 retention：未开始 | ✅ CrossMap-4 离散/连续<br>Seen-10 retention：未开始 |
| `exp34_gen` 50-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120<br>Seen-10 retention：未开始 | ✅ CrossMap-4 离散/连续<br>Seen-10 retention：未开始 |
| `exp34_gen` 20-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120<br>Seen-10 retention：未开始 | ✅ CrossMap-4 离散/连续<br>Seen-10 retention：未开始 |
| `exp34_gen` 10-shot | ✅ step 400 | ✅ CrossMap-4：离散 8,000 + 连续 5,120<br>Seen-10 retention：未开始 | ✅ CrossMap-4 离散/连续<br>Seen-10 retention：未开始 |
| `exp34_loc` 100-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp34_loc` 50-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp34_loc` 20-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
| `exp34_loc` 10-shot | ✅ step 400 | ✅ CrossMap-4 8,000 + Seen retention 20,000 | ✅ 两项均完成 |
|---|---|---|---|
| `exp36_cs_office` 100-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_golden` 100-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_palacio` 100-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_vertigo` 100-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_cs_office` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_gen_de_golden` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_gen_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_gen_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_loc_cs_office` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_loc_de_golden` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_loc_de_palacio` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_loc_de_vertigo` 100-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
|---|---|---|---|
| `exp36_cs_office` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_golden` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_palacio` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_vertigo` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_cs_office` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_gen_de_golden` 50-shot | ✅ step 400 | CrossMap-4 generation discrete+continuous<br>Seen-10 retention generation discrete+continuous | CrossMap discrete<br>CrossMap continuous<br>Seen retention discrete<br>Seen retention continuous |
| `exp36_gen_de_palacio` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_vertigo` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_cs_office` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_loc_de_golden` 50-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_palacio` 50-shot | ✅ step 400 | CrossMap-4 localization<br>Seen-10 retention localization | CrossMap localization<br>Seen retention localization |
| `exp36_loc_de_vertigo` 50-shot | 未开始 | 未开始 | 未开始 |
|---|---|---|---|
| `exp36_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_cs_office` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_golden` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_palacio` 20-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_vertigo` 20-shot | 未开始 | 未开始 | 未开始 |
|---|---|---|---|
| `exp36_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_gen_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_cs_office` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_golden` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_palacio` 10-shot | 未开始 | 未开始 | 未开始 |
| `exp36_loc_de_vertigo` 10-shot | 未开始 | 未开始 | 未开始 |
|---|---|---|---|



# ablation 3 maps表

2026-10-30 新增 B2_S3_J8（P4＋U2，attention=false）时仅有设计与 validation／test 占位；这是当日快照。2026-11-01 同步的实测数值见[独立 B2 消融补充](#b2-ablation-summary-20261030)。

`exp32_3maps_*` 按 L0、G0、J0～J4、补充对照 J5～J10、原第二轮 A0～A3、再到新增 A1_J8～A3_J8 的顺序加入各自有效任务表。固定地图为 `de_ancient`、`de_dust2`、`de_nuke`；S3 为 `[0,3000,5400,8400] -> [2,5,10,20]` 的分段线性 αloc 调度。原第二轮 A0～A3 保留 J4 默认配方；2026-10-14 新增 A1_J8～A3_J8 固定 J8 配方且独立输出。纳入 J9/J10 后，[2026-10-13 结果分析](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#j9-j10-results-20261013)仍建议 J8 作为新辅助组主配方，J9 保留为定位优先候选；实际运行以训练保存的 `resolved_config.yaml` 为准。

新 J8 辅助三组的定位、离散、连续结果当前均为空白占位，状态为已配置／未启动、待评测。`validation` 列区分筛选结果与测试报告；新实验的定位、离散生成行预留 `seen_validation` 结果，连续生成仅预留 `seen_continuous` 测试结果。`checkpoint steps` 是该行对应的单一 checkpoint：标注“计划”的步数未确认执行，连续测试的步数须在 validation 后选定。评测其他 checkpoint 或补充离散测试时，在对应实验旁新增明确 split／step 的结果行，分别记录，不覆盖既有结果。同一联合模型的定位／生成使用同一 checkpoint，比较时匹配步数；旧 exp31 指标原样保留，validation 列暂记 `—`，不与新 validation 数值直接比较。

J5～J8 的历史同一步数对照、逐图例外及当日后续执行计划见 [2026-10-12 结果分析](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#j5-j8-results-20261012)。


## 定位

| Setting | Task | Experiment | Shot/map | ablation | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | validation | checkpoint steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Seen-3 ablation | Localization | exp31_3maps | dust2, ancient, nuke | aux-loc loss + perception loss |  |  |  |  | — | - |
| Seen-3 ablation | Localization | exp31_1_3maps | dust2, ancient, nuke | - - | 83.911 | 3.887 | 3.251 | 35.434 | — | 5850 |
| Seen-3 ablation | Localization | exp31_2_3maps | dust2, ancient, nuke | aux-loc loss | 93.755 | 4.259 | 3.009 | 40.475 | — | 5900 |
| Seen-3 ablation | Localization | exp31_3_3maps | dust2, ancient, nuke | perception loss | 82.114 | 3.900 | 3.314 | 35.501 | — | 5900 |
| Seen-3 validation | Localization | exp32_3maps_loc_single | dust2, ancient, nuke | L0：定位单任务；无辅助 | 72.732 | 4.412 | 3.278 | 42.125 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_loc_single | dust2, ancient, nuke | L0：定位单任务；无辅助 | 48.459 | 2.878 | 2.818 | 26.235 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_original | dust2, ancient, nuke | J0：joint-only；原 LR；αloc=S3 | 131.159 | 8.237 | 3.325 | 59.230 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_original | dust2, ancient, nuke | J0：joint-only；原 LR；αloc=S3 | 82.517 | 4.018 | 2.752 | 34.065 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_headlr | dust2, ancient, nuke | J1：joint-only；定位私有 LR 对齐；αloc=S3 | 128.640 | 6.646 | 3.393 | 58.310 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_headlr | dust2, ancient, nuke | J1：joint-only；定位私有 LR 对齐；αloc=S3 | 88.997 | 4.515 | 3.060 | 33.879 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_sharedlr | dust2, ancient, nuke | J2：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=S3 | 149.063 | 7.017 | 4.763 | 62.676 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_sharedlr | dust2, ancient, nuke | J2：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=S3 | 89.496 | 4.512 | 2.557 | 37.365 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc | dust2, ancient, nuke | J3：joint-only；定位私有 LR 对齐；αloc=2 | 112.408 | 4.857 | 2.752 | 55.544 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc | dust2, ancient, nuke | J3：joint-only；定位私有 LR 对齐；αloc=2 | 67.537 | 3.412 | 2.626 | 35.161 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_sharedlr_constloc | dust2, ancient, nuke | J4：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=2 | 140.667 | 6.547 | 3.936 | 57.089 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_sharedlr_constloc | dust2, ancient, nuke | J4：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=2 | 87.428 | 4.485 | 2.918 | 36.186 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_original_constloc | dust2, ancient, nuke | J5：joint-only；原定位头 LR；αloc=2 | 126.706 | 6.907 | 2.793 | 61.391 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_original_constloc | dust2, ancient, nuke | J5：joint-only；原定位头 LR；αloc=2 | 82.201 | 4.131 | 2.666 | 35.574 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc1 | dust2, ancient, nuke | J6：joint-only；定位私有 LR 对齐；αloc=1 | 134.598 | 6.843 | 2.803 | 60.586 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc1 | dust2, ancient, nuke | J6：joint-only；定位私有 LR 对齐；αloc=1 | 89.705 | 4.530 | 2.736 | 36.633 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc4 | dust2, ancient, nuke | J7：joint-only；定位私有 LR 对齐；αloc=4 | 88.933 | 4.986 | 2.710 | 39.281 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc4 | dust2, ancient, nuke | J7：joint-only；定位私有 LR 对齐；αloc=4 | 57.560 | 3.188 | 2.588 | 28.979 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc8 | dust2, ancient, nuke | J8：joint-only；定位私有 LR 对齐；αloc=8 | 67.344 | 4.415 | 2.504 | 35.516 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc8 | dust2, ancient, nuke | J8：joint-only；定位私有 LR 对齐；αloc=8 | 47.113 | 2.665 | 2.548 | 25.070 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc12 | dust2, ancient, nuke | J9：joint-only；定位私有 LR 对齐；αloc=12 | 61.070 | 3.985 | 4.047 | 33.191 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc12 | dust2, ancient, nuke | J9：joint-only；定位私有 LR 对齐；αloc=12 | 44.067 | 2.907 | 2.421 | 24.334 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc16 | dust2, ancient, nuke | J10：joint-only；定位私有 LR 对齐；αloc=16 | 130.298 | 6.710 | 5.462 | 57.129 | seen_validation | 2400 |
| Seen-3 validation | Localization | exp32_3maps_joint_constloc16 | dust2, ancient, nuke | J10：joint-only；定位私有 LR 对齐；αloc=16 | 82.350 | 4.247 | 2.769 | 36.257 | seen_validation | 5850 |
| Seen-3 validation | Localization | exp32_3maps_aux_control | dust2, ancient, nuke | A0：joint-only；aux-loc 关＋perception 关 |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Localization | exp32_3maps_perception | dust2, ancient, nuke | A1：joint＋perception；aux-loc 关 |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Localization | exp32_3maps_auxloc | dust2, ancient, nuke | A2：joint＋aux-loc；perception 关 |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Localization | exp32_3maps_both | dust2, ancient, nuke | A3：joint＋aux-loc＋perception |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Localization | exp32_3maps_perception_j8 | dust2, ancient, nuke | A1_J8：joint＋perception；aux-loc 关；J8 配方 |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Localization | exp32_3maps_auxloc_j8 | dust2, ancient, nuke | A2_J8：joint＋aux-loc；perception 关；J8 配方 |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Localization | exp32_3maps_both_j8 | dust2, ancient, nuke | A3_J8：joint＋aux-loc＋perception；J8 配方 |  |  |  |  | seen_validation（待评测） | 3600（计划） |


| Seen-3 validation | Localization | exp32_3maps_stage2_control_j8 | dust2, ancient, nuke | S0_J8：joint；Stage 2 无辅助等预算对照 |  |  |  |  | seen_validation（待评测） | 600（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_control_j8 | dust2, ancient, nuke | S0_J8：joint；Stage 2 无辅助等预算对照 |  |  |  |  | seen_validation（待评测） | 1200（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_control_j8 | dust2, ancient, nuke | S0_J8：joint；Stage 2 无辅助等预算对照 |  |  |  |  | seen_validation（待评测） | 1800（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_perception_j8 | dust2, ancient, nuke | S1_J8：joint＋perception；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 600（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_perception_j8 | dust2, ancient, nuke | S1_J8：joint＋perception；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 1200（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_perception_j8 | dust2, ancient, nuke | S1_J8：joint＋perception；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 1800（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_auxloc_j8 | dust2, ancient, nuke | S2_J8：joint＋aux-loc；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 600（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_auxloc_j8 | dust2, ancient, nuke | S2_J8：joint＋aux-loc；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 1200（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_auxloc_j8 | dust2, ancient, nuke | S2_J8：joint＋aux-loc；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 1800（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_both_j8 | dust2, ancient, nuke | S3_J8：joint＋aux-loc＋perception；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 600（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_both_j8 | dust2, ancient, nuke | S3_J8：joint＋aux-loc＋perception；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 1200（Stage 2 local，计划） |
| Seen-3 validation | Localization | exp32_3maps_stage2_both_j8 | dust2, ancient, nuke | S3_J8：joint＋aux-loc＋perception；Stage 2 J8@5850 初始化 |  |  |  |  | seen_validation（待评测） | 1800（Stage 2 local，计划） |

### S0_L0 单任务续训（2026-10-30 同步）

| Setting | Task | Experiment | Shot/map | ablation | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | validation | checkpoint steps |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Seen-3 validation | Localization | exp32_3maps_stage2_control_l0 | dust2, ancient, nuke | S0_L0：L0@5850 初始化；Stage 2 定位单任务无辅助 | 46.478 | 2.826 | 2.745 | 25.307 | seen_validation | 1200（Stage 2 local；累计7050） |
| Seen-3 validation | Localization | exp32_3maps_stage2_control_l0 | dust2, ancient, nuke | S0_L0：L0@5850 初始化；Stage 2 定位单任务无辅助 | 45.651 | 2.821 | 2.358 | 24.979 | seen_validation | 1800（Stage 2 local；累计7650） |
| Seen-3 test | Localization | exp32_3maps_stage2_control_l0 | dust2, ancient, nuke | S0_L0：L0@5850 初始化；Stage 2 定位单任务无辅助 | 57.206 | 3.057 | 2.580 | 30.950 | seen_discrete_test（test） | 1800（Stage 2 local；累计7650） |

源指标：validation1200 [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_control_l0/checkpoint-1200/validation/seed_42/loc_metrics.json)、validation1800 [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_control_l0/checkpoint-1800/validation/seed_42/loc_metrics.json)、test1800 [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_control_l0/checkpoint-1800/test/seed_42/loc_metrics.json)；均为三图等权 `metrics_macro_map`，seed=42，validation 每图500、test 每图2,000条。生成任务为 N/A。完整进度、比较说明与最新 test 总表见[主实验文档](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#single-stage2-controls-20261030)。

## 离散生成

| Setting | Task | Experiment | Shot/map | ablation | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | validation | checkpoint steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Seen-3 ablation | Discrete generation | exp31_3maps | dust2, ancient, nuke | aux-loc loss + perception loss |  |  |  |  |  | — | - |
| Seen-3 ablation | Discrete generation | exp31_1_3maps | dust2, ancient, nuke | - - | 14.304 | 0.407 | 0.624 | 0.516 | 29.829 | — | 5850 |
| Seen-3 ablation | Discrete generation | exp31_2_3maps | dust2, ancient, nuke | aux-loc loss | 14.291 | 0.405 | 0.623 | 0.516 | 30.130 | — | 5900 |
| Seen-3 ablation | Discrete generation | exp31_3_3maps | dust2, ancient, nuke | perception loss | 14.616 | 0.409 | 0.607 | 0.527 | 30.430 | — | 5900 |
| Seen-3 validation | Discrete generation | exp32_3maps_gen_single | dust2, ancient, nuke | G0：生成单任务；无辅助 | 13.284 | 0.391 | 0.665 | 0.499 | 75.977 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_gen_single | dust2, ancient, nuke | G0：生成单任务；无辅助 | 14.402 | 0.418 | 0.613 | 0.520 | 62.252 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_original | dust2, ancient, nuke | J0：joint-only；原 LR；αloc=S3 | 13.521 | 0.407 | 0.652 | 0.497 | 67.498 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_original | dust2, ancient, nuke | J0：joint-only；原 LR；αloc=S3 | 14.592 | 0.425 | 0.606 | 0.523 | 61.835 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_headlr | dust2, ancient, nuke | J1：joint-only；定位私有 LR 对齐；αloc=S3 | 13.529 | 0.398 | 0.653 | 0.499 | 69.793 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_headlr | dust2, ancient, nuke | J1：joint-only；定位私有 LR 对齐；αloc=S3 | 14.515 | 0.421 | 0.608 | 0.524 | 60.631 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_sharedlr | dust2, ancient, nuke | J2：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=S3 | 13.238 | 0.396 | 0.666 | 0.496 | 66.886 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_sharedlr | dust2, ancient, nuke | J2：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=S3 | 14.208 | 0.421 | 0.632 | 0.508 | 64.740 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc | dust2, ancient, nuke | J3：joint-only；定位私有 LR 对齐；αloc=2 | 13.267 | 0.389 | 0.659 | 0.501 | 68.925 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc | dust2, ancient, nuke | J3：joint-only；定位私有 LR 对齐；αloc=2 | 14.464 | 0.419 | 0.609 | 0.522 | 65.062 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_sharedlr_constloc | dust2, ancient, nuke | J4：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=2 | 13.308 | 0.387 | 0.669 | 0.495 | 65.208 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_sharedlr_constloc | dust2, ancient, nuke | J4：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=2 | 14.111 | 0.413 | 0.630 | 0.512 | 63.516 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_original_constloc | dust2, ancient, nuke | J5：joint-only；原定位头 LR；αloc=2 | 13.361 | 0.397 | 0.657 | 0.498 | 65.284 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_original_constloc | dust2, ancient, nuke | J5：joint-only；原定位头 LR；αloc=2 | 14.518 | 0.420 | 0.607 | 0.524 | 59.700 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc1 | dust2, ancient, nuke | J6：joint-only；定位私有 LR 对齐；αloc=1 | 13.356 | 0.409 | 0.655 | 0.496 | 68.972 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc1 | dust2, ancient, nuke | J6：joint-only；定位私有 LR 对齐；αloc=1 | 14.607 | 0.421 | 0.603 | 0.528 | 61.294 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc4 | dust2, ancient, nuke | J7：joint-only；定位私有 LR 对齐；αloc=4 | 13.227 | 0.405 | 0.659 | 0.496 | 64.834 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc4 | dust2, ancient, nuke | J7：joint-only；定位私有 LR 对齐；αloc=4 | 14.198 | 0.416 | 0.621 | 0.518 | 62.457 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc8 | dust2, ancient, nuke | J8：joint-only；定位私有 LR 对齐；αloc=8 | 13.420 | 0.396 | 0.660 | 0.495 | 65.713 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc8 | dust2, ancient, nuke | J8：joint-only；定位私有 LR 对齐；αloc=8 | 14.222 | 0.416 | 0.623 | 0.517 | 61.569 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc12 | dust2, ancient, nuke | J9：joint-only；定位私有 LR 对齐；αloc=12 | 13.204 | 0.388 | 0.674 | 0.495 | 65.632 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc12 | dust2, ancient, nuke | J9：joint-only；定位私有 LR 对齐；αloc=12 | 14.087 | 0.416 | 0.638 | 0.501 | 64.164 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc16 | dust2, ancient, nuke | J10：joint-only；定位私有 LR 对齐；αloc=16 | 13.346 | 0.393 | 0.666 | 0.496 | 69.046 | seen_validation | 2400 |
| Seen-3 validation | Discrete generation | exp32_3maps_joint_constloc16 | dust2, ancient, nuke | J10：joint-only；定位私有 LR 对齐；αloc=16 | 14.498 | 0.421 | 0.612 | 0.521 | 62.497 | seen_validation | 5850 |
| Seen-3 validation | Discrete generation | exp32_3maps_aux_control | dust2, ancient, nuke | A0：joint-only；aux-loc 关＋perception 关 |  |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_perception | dust2, ancient, nuke | A1：joint＋perception；aux-loc 关 |  |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_auxloc | dust2, ancient, nuke | A2：joint＋aux-loc；perception 关 |  |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_both | dust2, ancient, nuke | A3：joint＋aux-loc＋perception |  |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_perception_j8 | dust2, ancient, nuke | A1_J8：joint＋perception；aux-loc 关；J8 配方 |  |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_auxloc_j8 | dust2, ancient, nuke | A2_J8：joint＋aux-loc；perception 关；J8 配方 |  |  |  |  |  | seen_validation（待评测） | 3600（计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_both_j8 | dust2, ancient, nuke | A3_J8：joint＋aux-loc＋perception；J8 配方 |  |  |  |  |  | seen_validation（待评测） | 3600（计划） |


| Seen-3 validation | Discrete generation | exp32_3maps_stage2_control_j8 | dust2, ancient, nuke | S0_J8：joint；Stage 2 无辅助等预算对照 |  |  |  |  |  | seen_validation（待评测） | 600（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_control_j8 | dust2, ancient, nuke | S0_J8：joint；Stage 2 无辅助等预算对照 |  |  |  |  |  | seen_validation（待评测） | 1200（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_control_j8 | dust2, ancient, nuke | S0_J8：joint；Stage 2 无辅助等预算对照 |  |  |  |  |  | seen_validation（待评测） | 1800（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_perception_j8 | dust2, ancient, nuke | S1_J8：joint＋perception；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 600（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_perception_j8 | dust2, ancient, nuke | S1_J8：joint＋perception；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 1200（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_perception_j8 | dust2, ancient, nuke | S1_J8：joint＋perception；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 1800（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_auxloc_j8 | dust2, ancient, nuke | S2_J8：joint＋aux-loc；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 600（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_auxloc_j8 | dust2, ancient, nuke | S2_J8：joint＋aux-loc；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 1200（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_auxloc_j8 | dust2, ancient, nuke | S2_J8：joint＋aux-loc；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 1800（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_both_j8 | dust2, ancient, nuke | S3_J8：joint＋aux-loc＋perception；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 600（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_both_j8 | dust2, ancient, nuke | S3_J8：joint＋aux-loc＋perception；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 1200（Stage 2 local，计划） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_both_j8 | dust2, ancient, nuke | S3_J8：joint＋aux-loc＋perception；Stage 2 J8@5850 初始化 |  |  |  |  |  | seen_validation（待评测） | 1800（Stage 2 local，计划） |

### S0_G0 单任务续训（2026-10-31 同步）

| Setting | Task | Experiment | Shot/map | ablation | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | validation | checkpoint steps |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_control_g0 | dust2, ancient, nuke | S0_G0：G0@5850 初始化；Stage 2 生成单任务无辅助 | 14.568 | 0.421 | 0.607 | 0.523 | 62.737 | seen_validation | 1200（Stage 2 local；累计7050） |
| Seen-3 validation | Discrete generation | exp32_3maps_stage2_control_g0 | dust2, ancient, nuke | S0_G0：G0@5850 初始化；Stage 2 生成单任务无辅助 | 14.622 | 0.420 | 0.604 | 0.526 | 62.401 | seen_validation | 1800（Stage 2 local；累计7650） |
| Seen-3 test | Discrete generation | exp32_3maps_stage2_control_g0 | dust2, ancient, nuke | S0_G0：G0@5850 初始化；Stage 2 生成单任务无辅助 | 14.258 | 0.401 | 0.622 | 0.521 | 36.780 | seen_discrete_test（test） | 1800（Stage 2 local；累计7650） |

三图等权 `metrics_macro_map`，seed=42；validation 每图500（共1,500），test 每图2,000（共6,000）。源 JSON：[validation1200](outputs_eval/benchmark_v2/exp32_3maps_stage2_control_g0/checkpoint-1200/validation/seed_42/discrete/summary.json)、[validation1800](outputs_eval/benchmark_v2/exp32_3maps_stage2_control_g0/checkpoint-1800/validation/seed_42/discrete/summary.json)、[test1800](outputs_eval/benchmark_v2/exp32_3maps_stage2_control_g0/checkpoint-1800/test/seed_42/discrete/summary.json)。相对 G0@5850，validation1800 五项中除 FID 外均改善，test 五项均改善。完整进度及主文档结果见[Stage 2 单任务对照](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#single-stage2-controls-20261030)。

## 连续生成

| Setting | Task | Experiment | Shot/map | ablation | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | FVD↓ | validation | checkpoint steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Seen-3 ablation | Continuous generation | exp31_3maps | dust2, ancient, nuke | aux-loc loss + perception loss |  |  |  |  |  |  | — | - |
| Seen-3 ablation | Continuous generation | exp31_1_3maps | dust2, ancient, nuke | - - | 14.761 | 0.423 | 0.616 | 34.344 | 39.686 | 886.168 | — | 5850 |
| Seen-3 ablation | Continuous generation | exp31_2_3maps | dust2, ancient, nuke | aux-loc loss | 14.811 | 0.424 | 0.611 | 32.870 | 38.398 | 831.028 | — | 5900 |
| Seen-3 ablation | Continuous generation | exp31_3_3maps | dust2, ancient, nuke | perception loss | 15.303 | 0.433 | 0.586 | 32.357 | 37.824 | 836.473 | — | 5900 |
| Seen-3 test | Continuous generation | exp32_3maps_gen_single | dust2, ancient, nuke | G0：生成单任务；无辅助 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_original | dust2, ancient, nuke | J0：joint-only；原 LR；αloc=S3 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_headlr | dust2, ancient, nuke | J1：joint-only；定位私有 LR 对齐；αloc=S3 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_sharedlr | dust2, ancient, nuke | J2：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=S3 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_constloc | dust2, ancient, nuke | J3：joint-only；定位私有 LR 对齐；αloc=2 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_sharedlr_constloc | dust2, ancient, nuke | J4：joint-only；定位私有 LR 对齐＋共享 LLM LR=3e-5；αloc=2 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_original_constloc | dust2, ancient, nuke | J5：joint-only；原定位头 LR；αloc=2 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_constloc1 | dust2, ancient, nuke | J6：joint-only；定位私有 LR 对齐；αloc=1 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_constloc4 | dust2, ancient, nuke | J7：joint-only；定位私有 LR 对齐；αloc=4 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_constloc8 | dust2, ancient, nuke | J8：joint-only；定位私有 LR 对齐；αloc=8 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_constloc12 | dust2, ancient, nuke | J9：joint-only；定位私有 LR 对齐；αloc=12 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_joint_constloc16 | dust2, ancient, nuke | J10：joint-only；定位私有 LR 对齐；αloc=16 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_aux_control | dust2, ancient, nuke | A0：joint-only；aux-loc 关＋perception 关 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_perception | dust2, ancient, nuke | A1：joint＋perception；aux-loc 关 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_auxloc | dust2, ancient, nuke | A2：joint＋aux-loc；perception 关 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_both | dust2, ancient, nuke | A3：joint＋aux-loc＋perception |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_perception_j8 | dust2, ancient, nuke | A1_J8：joint＋perception；aux-loc 关；J8 配方 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_auxloc_j8 | dust2, ancient, nuke | A2_J8：joint＋aux-loc；perception 关；J8 配方 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |
| Seen-3 test | Continuous generation | exp32_3maps_both_j8 | dust2, ancient, nuke | A3_J8：joint＋aux-loc＋perception；J8 配方 |  |  |  |  |  |  | —（仅 test；待筛选后报告） | 待 validation 选定 |



### S0_G0 单任务续训（2026-10-31 同步）

| Setting | Task | Experiment | Shot/map | ablation | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | FVD↓ | validation | checkpoint steps |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Seen-3 test | Continuous generation | exp32_3maps_stage2_control_g0 | dust2, ancient, nuke | S0_G0：G0@5850 初始化；Stage 2 生成单任务无辅助 | 15.146 | 0.432 | 0.583 | 32.806 | 37.996 | 862.918 | seen_continuous（test） | 1800（Stage 2 local；累计7650） |

三图等权 `metrics_macro_map`，seed=42；test 为每图20条轨迹、每条64帧（共60条轨迹／3,840帧）。表中六项外，Boundary_F1=0.542、Optical_Flow_EPE=16.170。源 JSON：[test1800 continuous summary](outputs_eval/benchmark_v2/exp32_3maps_stage2_control_g0/checkpoint-1800/test/seed_42/continuous/summary.json)。相对 G0@5850，八项指标均改善；完整进度及主文档结果见[Stage 2 单任务对照](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#single-stage2-controls-20261030)。

# csgo benchmark v2 主表

## 定位
<!-- | Seen-10 | Localization | exp31 | - | 195.048 | 10.489 | 4.344 | 79.060 | 6000 | -->
<!-- | Seen-10 | Localization | exp32 | - | 211.975 | 10.766 | 4.652 | 81.964 | 6000 | -->
| Setting | Task | Experiment | Shot/map | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Seen-10 | Localization | exp31_loc | - | 79.294 | 4.134 | 2.708 | 36.173 | 19500 |
| Seen-10 | Localization | exp31_1 | - | 82.750 | 4.354 | 2.969 | 37.130 | 19500 |
| Seen-10 | Localization | exp31 | - | 99.0970 | 4.9604 | 3.2192 | 45.1100 | 19550 |
| CrossMap-4 zero-shot | Localization | exp31_1 | - | 330.421 | 17.483 | 3.895 | 89.844 | 19500 |
| CrossMap-4 zero-shot | Localization | exp31 | - | 328.6658 | 17.9069 | 3.9495 | 90.0281 | 19550 |
| CrossMap-4 zero-shot | Localization | exp31_loc | - |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Seen-10 | Localization | exp32 | - | 123.194 | 6.38 | 3.352 | 55.449 | 19500 |
| CrossMap-4 zero-shot | Localization | exp32 | - | 340.912 | 18.794 | 4.188 | 87.616 | 19500 |
| Seen-10 | Localization | exp32_loc | - | 48.961 | 3.006 | 2.970 | 26.047 | 19500 |
| CrossMap-4 zero-shot | Localization | exp32_loc | - |  |  |  |  | - |
| Seen-10 | Localization | exp32_1 | - | 63.0377 | 3.5604 | 3.0882 | 31.6713 | 19500 |
| CrossMap-4 zero-shot | Localization | exp32_1 | - |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp35_cs_office | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_golden | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_palacio | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_vertigo | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_1_cs_office | 100 | 98.629 | 1.712 | 4.728 | 73.606 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_golden | 100 | 170.524 | 11.668 | 2.477 | 71.589 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_palacio | 100 | 210.064 | 12.807 | 3.128 | 66.018 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_vertigo | 100 | 165.351 | 16.093 | 4.032 | 76.628 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_cs_office | 100 | 93.146 | 1.687 | 4.160 | 69.665 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_golden | 100 | 175.481 | 11.026 | 3.084 | 72.338 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_palacio | 100 | 218.156 | 12.130 | 3.304 | 65.144 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_vertigo | 100 | 164.352 | 16.383 | 3.916 | 72.533 | 400 |
| Seen-10 retention | Localization | exp35_cs_office | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_golden | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_palacio | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_vertigo | 100 |  |  |  |  | - |
| Seen-10 retention | Localization | exp36_1_cs_office | 100 | 253.912 | 15.469 | 6.729 | 66.152 | 400 |
| Seen-10 retention | Localization | exp36_1_de_golden | 100 | 203.442 | 11.969 | 3.226 | 64.464 | 400 |
| Seen-10 retention | Localization | exp36_1_de_palacio | 100 | 213.717 | 13.140 | 4.178 | 63.905 | 400 |
| Seen-10 retention | Localization | exp36_1_de_vertigo | 100 | 215.108 | 15.277 | 4.896 | 66.513 | 400 |
| Seen-10 retention | Localization | exp35_loc_cs_office | 100 | 256.115 | 17.999 | 5.421 | 68.599 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_golden | 100 | 216.713 | 13.237 | 3.708 | 68.459 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_palacio | 100 | 217.588 | 12.125 | 4.046 | 59.172 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_vertigo | 100 | 208.859 | 13.430 | 4.537 | 59.888 | 400 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp35_cs_office | 50 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_golden | 50 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_palacio | 50 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_vertigo | 50 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_1_cs_office | 50 | 122.239 | 2.100 | 4.750 | 81.182 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_golden | 50 | 175.451 | 12.672 | 2.785 | 80.600 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_palacio | 50 | 258.732 | 14.284 | 3.558 | 73.738 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_vertigo | 50 | 167.365 | 17.538 | 3.446 | 81.584 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_cs_office | 50 | 121.609 | 2.313 | 4.390 | 84.044 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_golden | 50 | 182.041 | 13.299 | 3.023 | 82.515 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_palacio | 50 | 239.941 | 13.969 | 3.574 | 72.311 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_de_vertigo | 50 | 166.369 | 16.331 | 3.869 | 76.606 | 400 |
| Seen-10 retention | Localization | exp35_cs_office | 50 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_golden | 50 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_palacio | 50 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_vertigo | 50 |  |  |  |  | - |
| Seen-10 retention | Localization | exp36_1_cs_office | 50 | 287.580 | 18.498 | 8.212 | 60.659 | 400 |
| Seen-10 retention | Localization | exp36_1_de_golden | 50 | 198.294 | 12.191 | 3.734 | 68.007 | 400 |
| Seen-10 retention | Localization | exp36_1_de_palacio | 50 | 231.973 | 16.715 | 3.883 | 61.935 | 400 |
| Seen-10 retention | Localization | exp36_1_de_vertigo | 50 | 229.513 | 11.275 | 5.039 | 61.304 | 400 |
| Seen-10 retention | Localization | exp35_loc_cs_office | 50 | 267.171 | 17.694 | 7.181 | 68.238 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_golden | 50 | 232.139 | 18.689 | 3.783 | 72.837 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_palacio | 50 | 228.142 | 16.225 | 3.717 | 63.418 | 400 |
| Seen-10 retention | Localization | exp35_loc_de_vertigo | 50 | 221.817 | 13.054 | 3.903 | 63.401 | 400 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp35_cs_office | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_golden | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_palacio | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_vertigo | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_1_cs_office | 20 | 140.899 | 2.139 | 4.981 | 86.700 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_golden | 20 | 205.050 | 14.138 | 2.126 | 82.885 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_palacio | 20 | 263.819 | 14.510 | 3.523 | 79.928 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_vertigo | 20 | 179.212 | 20.513 | 3.026 | 84.123 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_cs_office | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc_de_golden | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc_de_palacio | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc_de_vertigo | 20 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_cs_office | 20 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_golden | 20 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_palacio | 20 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_vertigo | 20 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_loc_de_vertigo | 20 |  |  |  |  | - |
| Seen-10 retention | Localization | exp36_1_cs_office | 20 | 261.939 | 17.207 | 6.856 | 69.089 | 400 |
| Seen-10 retention | Localization | exp36_1_de_golden | 20 | 225.539 | 12.908 | 3.160 | 75.435 | 400 |
| Seen-10 retention | Localization | exp36_1_de_palacio | 20 | 225.915 | 16.806 | 3.802 | 69.596 | 400 |
| Seen-10 retention | Localization | exp36_1_de_vertigo | 20 | 216.861 | 11.668 | 3.607 | 66.533 | 400 |
| Seen-10 retention | Localization | exp35_loc_cs_office | 20 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_loc_de_golden | 20 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_loc_de_palacio | 20 |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp35_cs_office | 10 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_golden | 10 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_palacio | 10 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_de_vertigo | 10 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_1_cs_office | 10 | 149.829 | 2.761 | 5.122 | 86.316 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_golden | 10 | 219.587 | 17.857 | 2.154 | 86.595 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_palacio | 10 | 267.508 | 14.773 | 2.765 | 88.412 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1_de_vertigo | 10 | 193.268 | 20.186 | 2.737 | 84.196 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc_cs_office | 10 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc_de_golden | 10 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc_de_palacio | 10 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc_de_vertigo | 10 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_cs_office | 10 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_golden | 10 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_palacio | 10 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_de_vertigo | 10 |  |  |  |  | - |
| Seen-10 retention | Localization | exp36_1_cs_office | 10 | 270.479 | 16.147 | 6.913 | 75.941 | 400 |
| Seen-10 retention | Localization | exp36_1_de_golden | 10 | 209.586 | 12.878 | 3.189 | 74.860 | 400 |
| Seen-10 retention | Localization | exp36_1_de_palacio | 10 | 229.112 | 15.516 | 3.162 | 80.364 | 400 |
| Seen-10 retention | Localization | exp36_1_de_vertigo | 10 | 214.614 | 17.543 | 3.145 | 65.920 | 400 |
| Seen-10 retention | Localization | exp35_loc_cs_office | 10 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_loc_de_golden | 10 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_loc_de_palacio | 10 |  |  |  |  | - |
| Seen-10 retention | Localization | exp35_loc_de_vertigo | 10 |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp35 | 100 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35 | 50 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35 | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35 | 10 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp36_1 | 100 | 161.142 | 10.570 | 3.591 | 71.960 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1 | 50 | 180.947 | 11.649 | 3.635 | 79.276 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1 | 20 | 197.245 | 12.825 | 3.414 | 83.409 | 400 |
| CrossMap-4 few-shot | Localization | exp36_1 | 10 | 207.548 | 13.894 | 3.194 | 86.380 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc | 100 | 162.784 | 10.307 | 3.616 | 69.920 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc | 50 | 177.490 | 11.478 | 3.714 | 78.869 | 400 |
| CrossMap-4 few-shot | Localization | exp35_loc | 20 |  |  |  |  | - |
| CrossMap-4 few-shot | Localization | exp35_loc | 10 |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|


## 离散生成
<!-- | Seen-10 | Discrete generation | exp31 | - | 13.212 | 0.4054 | 0.6425 | 0.4972 | 36.066 | 6000 | -->
<!-- | Seen-10 | Discrete generation | exp32 | - | 12.995 | 0.3974 | 0.6583 | 0.4889 | 44.101 | 6000 | -->
| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Discrete generation | exp31_gen | - | 14.561 | 0.4268 | 0.5888 | 0.5258 | 28.629 | 19500 |
| Seen-10 | Discrete generation | exp31_1 | - | 14.954 | 0.4319 | 0.5725 | 0.5375 | 27.949 | 19500 |
| Seen-10 | Discrete generation | exp31 | - | 14.6271 | 0.4259 | 0.5847 | 0.5294 | 28.3035 | 19550 |
| CrossMap-4 zero-shot | Discrete generation | exp31_gen | - | 12.288 | 0.4380 | 0.7358 | 0.4542 | 101.199 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp31_1 | - | 11.747 | 0.4156 | 0.7400 | 0.4522 | 96.250 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp31 | - | 12.0959 | 0.4237 | 0.7352 | 0.4483 | 86.8346 | 19550 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Seen-10 | Discrete generation | exp32 | - | 14.1528 | 0.4141 | 0.6035 | 0.5229 | 36.6005 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp32 | - | 11.5381 | 0.4075 | 0.7396 | 0.4471 | 88.0380 | 19500 |
| Seen-10 | Discrete generation | exp32_gen | - | 13.965 | 0.4113 | 0.6106 | 0.5163 | 35.584 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp32_gen | - | 11.081 | 0.3690 | 0.7210 | 0.4592 | 79.887 | 19500 |
| Seen-10 | Discrete generation | exp32_1 | - | 13.9070 | 0.4072 | 0.6145 | 0.5151 | 36.2159 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp32_1 | - |  |  |  |  |  | 19500 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp35_cs_office | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_golden | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_palacio | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_vertigo | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_1_cs_office | 100 | 14.906 | 0.6103 | 0.5904 | 0.4763 | 34.585 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_golden | 100 | 15.541 | 0.3812 | 0.5913 | 0.5435 | 38.245 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_palacio | 100 | 13.879 | 0.3779 | 0.6638 | 0.5376 | 39.222 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_vertigo | 100 | 12.845 | 0.4706 | 0.6344 | 0.4696 | 40.943 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 100 | 14.839 | 0.6121 | 0.5995 | 0.4686 | 34.197 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_golden | 100 | 15.506 | 0.3827 | 0.5924 | 0.5405 | 39.009 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_palacio | 100 | 13.630 | 0.3737 | 0.6682 | 0.5374 | 36.978 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_vertigo | 100 | 12.469 | 0.4603 | 0.6468 | 0.4609 | 39.447 | 400 |
| Seen-10 retention | Discrete generation | exp35_cs_office | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_golden | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_palacio | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_vertigo | 100 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp36_1_cs_office | 100 | 13.582 | 0.4683 | 0.6881 | 0.4811 | 104.002 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_golden | 100 | 13.546 | 0.3872 | 0.6272 | 0.5215 | 66.075 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_palacio | 100 | 13.746 | 0.4154 | 0.6514 | 0.5021 | 68.688 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_vertigo | 100 | 13.078 | 0.4424 | 0.6615 | 0.4778 | 73.753 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_cs_office | 100 | 13.499 | 0.4629 | 0.6930 | 0.4798 | 116.424 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_golden | 100 | 13.449 | 0.3882 | 0.6298 | 0.5171 | 67.519 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_palacio | 100 | 13.377 | 0.4094 | 0.6583 | 0.4973 | 67.946 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_vertigo | 100 | 12.719 | 0.4287 | 0.6695 | 0.4746 | 75.466 | 400 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp35_cs_office | 50 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_golden | 50 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_palacio | 50 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_vertigo | 50 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_1_cs_office | 50 | 14.086 | 0.6068 | 0.6216 | 0.4516 | 46.612 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_golden | 50 | 15.211 | 0.3760 | 0.6136 | 0.5346 | 51.181 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_palacio | 50 | 13.312 | 0.3794 | 0.6892 | 0.5218 | 52.357 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_vertigo | 50 | 12.377 | 0.4614 | 0.6544 | 0.4599 | 50.240 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 50 | 13.952 | 0.6020 | 0.6342 | 0.4448 | 45.005 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_golden | 50 | 15.174 | 0.3754 | 0.6066 | 0.5333 | 48.969 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_palacio | 50 | 13.221 | 0.3721 | 0.6932 | 0.5199 | 48.219 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_vertigo | 50 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_cs_office | 50 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_golden | 50 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_palacio | 50 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_vertigo | 50 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp36_1_cs_office | 50 | 13.295 | 0.4653 | 0.7033 | 0.4790 | 126.942 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_golden | 50 | 13.518 | 0.3976 | 0.6409 | 0.5195 | 75.333 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_palacio | 50 | 13.613 | 0.4332 | 0.6758 | 0.5001 | 92.391 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_vertigo | 50 | 13.036 | 0.4476 | 0.6646 | 0.4778 | 86.895 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_cs_office | 50 | 13.303 | 0.4660 | 0.7066 | 0.4739 | 129.509 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_golden | 50 | 13.362 | 0.3891 | 0.6363 | 0.5161 | 72.551 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_palacio | 50 | 13.426 | 0.4233 | 0.6768 | 0.4927 | 81.604 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_de_vertigo | 50 |  |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp35_cs_office | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_golden | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_palacio | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_vertigo | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_1_cs_office | 20 | 13.375 | 0.6028 | 0.6487 | 0.4283 | 66.517 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_golden | 20 | 15.031 | 0.3821 | 0.6224 | 0.5262 | 73.553 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_palacio | 20 | 12.977 | 0.3767 | 0.7112 | 0.5154 | 66.020 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_vertigo | 20 | 12.156 | 0.4711 | 0.6739 | 0.4507 | 73.408 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_golden | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_palacio | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_vertigo | 20 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_cs_office | 20 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_golden | 20 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_palacio | 20 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_vertigo | 20 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp36_1_cs_office | 20 | 13.119 | 0.4799 | 0.7242 | 0.4663 | 142.086 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_golden | 20 | 13.289 | 0.4103 | 0.6511 | 0.5223 | 92.416 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_palacio | 20 | 12.860 | 0.4229 | 0.7092 | 0.4918 | 122.822 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_vertigo | 20 | 12.656 | 0.4492 | 0.6836 | 0.4686 | 120.741 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_cs_office | 20 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_gen_de_golden | 20 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_gen_de_palacio | 20 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_gen_de_vertigo | 20 |  |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp35_cs_office | 10 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_golden | 10 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_palacio | 10 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_de_vertigo | 10 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_1_cs_office | 10 | 12.536 | 0.6121 | 0.6850 | 0.3967 | 120.828 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_golden | 10 | 14.485 | 0.3638 | 0.6353 | 0.5119 | 100.393 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_palacio | 10 | 12.187 | 0.3459 | 0.7179 | 0.5085 | 104.259 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_vertigo | 10 | 11.743 | 0.4756 | 0.6843 | 0.4507 | 129.216 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 10 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_golden | 10 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_palacio | 10 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen_de_vertigo | 10 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_cs_office | 10 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_golden | 10 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_palacio | 10 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_de_vertigo | 10 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_gen_de_vertigo | 10 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp36_1_cs_office | 10 | 12.201 | 0.4524 | 0.7367 | 0.4572 | 147.389 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_golden | 10 | 12.906 | 0.3970 | 0.6629 | 0.5098 | 126.722 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_palacio | 10 | 12.998 | 0.4054 | 0.7020 | 0.4918 | 143.218 | 400 |
| Seen-10 retention | Discrete generation | exp36_1_de_vertigo | 10 | 12.243 | 0.4434 | 0.7020 | 0.4752 | 158.646 | 400 |
| Seen-10 retention | Discrete generation | exp35_gen_cs_office | 10 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_gen_de_golden | 10 |  |  |  |  |  | - |
| Seen-10 retention | Discrete generation | exp35_gen_de_palacio | 10 |  |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp35 | 100 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35 | 50 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35 | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35 | 10 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp36_1 | 100 | 14.293 | 0.4600 | 0.6200 | 0.5067 | 38.249 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1 | 50 | 13.746 | 0.4559 | 0.6447 | 0.4920 | 50.098 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1 | 20 | 13.385 | 0.4581 | 0.6641 | 0.4802 | 69.874 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp36_1 | 10 | 12.738 | 0.4494 | 0.6806 | 0.4669 | 113.674 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen | 100 | 14.111 | 0.4572 | 0.6267 | 0.5018 | 37.408 | 400 |
| CrossMap-4 few-shot | Discrete generation | exp35_gen | 50 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen | 20 |  |  |  |  |  | - |
| CrossMap-4 few-shot | Discrete generation | exp35_gen | 10 |  |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|


## 连续生成
<!-- | Seen-10 | Continuous generation | exp31 | - | 13.255 | 0.4026 | 0.6417 | 40.134 | 46.190 | 1011.744 | 6000 | -->
<!-- | Seen-10 | Continuous generation | exp32 | - | 13.078 | 0.3953 | 0.6601 | 40.511 | 46.401 | 1058.158 | 6000 | -->
| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | FVD↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Continuous generation | exp31_gen | - | 15.164 | 0.4349 | 0.5622 | 31.409 | 38.033 | 737.197 | 19500 |
| Seen-10 | Continuous generation | exp31_1 | - | 15.571 | 0.4411 | 0.5446 | 28.879 | 35.688 | 674.180 | 19500 |
| Seen-10 | Continuous generation | exp31 | - | 15.1897 | 0.4330 | 0.5592 | 31.0365 | 37.6569 | 727.5069 | 19550 |
| CrossMap-4 zero-shot | Continuous generation | exp31_gen | - | 12.252 | 0.4291 | 0.7379 | 35.262 | 42.141 | 1131.397 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp31_1 | - | 11.826 | 0.4107 | 0.7421 | 37.947 | 44.492 | 1158.692 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp31 | - | 11.9599 | 0.4129 | 0.7360 | 38.1472 | 44.7851 | 122.9265 | 19550 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Continuous generation | exp32 | - | 14.5725 | 0.4190 | 0.5826 | 34.0925 | 40.3596 | 848.4586 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp32 | - | 11.5966 | 0.4018 | 0.7452 | 44.2626 | 50.0603 | 1255.7838 | 19500 |
| Seen-10 | Continuous generation | exp32_gen | - | 14.342 | 0.4162 | 0.5929 | 35.698 | 41.867 | 848.071 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp32_gen | - | 11.042 | 0.3628 | 0.7208 | 46.932 | 52.377 | 1349.984 | 19500 |
| Seen-10 | Continuous generation | exp32_1 | - | 14.2285 | 0.4100 | 0.5981 | 36.1777 | 42.2848 | 891.6025 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp32_1 | - |  |  |  |  |  |  | 19500 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp35_cs_office | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_golden | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_palacio | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_vertigo | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_1_cs_office | 100 | 15.044 | 0.5822 | 0.6040 | 18.660 | 26.817 | 445.689 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_golden | 100 | 15.563 | 0.3765 | 0.5816 | 18.594 | 24.810 | 707.997 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_palacio | 100 | 13.246 | 0.3536 | 0.6826 | 22.666 | 30.723 | 832.711 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_vertigo | 100 | 12.927 | 0.4623 | 0.6283 | 24.005 | 32.249 | 735.270 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_cs_office | 100 | 14.849 | 0.5840 | 0.6113 | 20.140 | 28.299 | 455.915 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_golden | 100 | 15.598 | 0.3767 | 0.5798 | 19.058 | 25.349 | 706.815 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_palacio | 100 | 13.158 | 0.3490 | 0.6859 | 24.469 | 32.308 | 775.390 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_vertigo | 100 | 12.166 | 0.4447 | 0.6455 | 24.464 | 32.945 | 730.773 | 400 |
| Seen-10 retention | Continuous generation | exp35_cs_office | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_golden | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_palacio | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_vertigo | 100 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp36_1_cs_office | 100 | 13.644 | 0.4671 | 0.6859 | 25.819 | 33.382 | 1604.541 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_golden | 100 | 13.636 | 0.3842 | 0.6255 | 23.042 | 30.843 | 890.939 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_palacio | 100 | 14.171 | 0.4231 | 0.6363 | 25.239 | 32.877 | 911.894 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_vertigo | 100 | 13.405 | 0.4445 | 0.6523 | 30.148 | 37.374 | 1159.175 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_cs_office | 100 | 13.619 | 0.4645 | 0.6864 | 25.670 | 33.345 | 1655.739 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_golden | 100 | 13.546 | 0.3867 | 0.6287 | 24.207 | 31.901 | 909.982 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_palacio | 100 | 13.783 | 0.4174 | 0.6501 | 26.768 | 34.261 | 993.103 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_vertigo | 100 | 13.070 | 0.4332 | 0.6642 | 31.365 | 38.412 | 1143.649 | 400 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp35_cs_office | 50 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_golden | 50 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_palacio | 50 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_vertigo | 50 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_1_cs_office | 50 | 13.434 | 0.5723 | 0.6559 | 20.370 | 28.521 | 615.408 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_golden | 50 | 15.350 | 0.3772 | 0.6054 | 17.216 | 23.258 | 864.206 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_palacio | 50 | 12.840 | 0.3641 | 0.7152 | 19.197 | 28.217 | 977.899 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_vertigo | 50 | 12.213 | 0.4485 | 0.6503 | 23.231 | 31.611 | 867.434 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_cs_office | 50 | 13.214 | 0.5588 | 0.6678 | 19.668 | 27.716 | 679.993 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_golden | 50 | 15.230 | 0.3736 | 0.5949 | 18.028 | 24.092 | 804.880 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_palacio | 50 | 12.531 | 0.3449 | 0.7141 | 23.095 | 31.315 | 930.480 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_vertigo | 50 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_cs_office | 50 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_golden | 50 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_palacio | 50 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_vertigo | 50 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp36_1_cs_office | 50 | 13.329 | 0.4650 | 0.7057 | 24.584 | 32.383 | 1729.202 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_golden | 50 | 13.578 | 0.3972 | 0.6418 | 21.846 | 29.934 | 1020.407 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_palacio | 50 | 13.918 | 0.4397 | 0.6657 | 22.906 | 31.142 | 1185.522 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_vertigo | 50 | 13.272 | 0.4495 | 0.6603 | 27.879 | 35.415 | 1282.012 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_cs_office | 50 | 13.407 | 0.4670 | 0.7045 | 26.690 | 34.130 | 1699.999 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_golden | 50 | 13.427 | 0.3894 | 0.6348 | 23.502 | 31.332 | 969.587 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_palacio | 50 | 13.751 | 0.4320 | 0.6672 | 25.390 | 33.194 | 1120.057 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_de_vertigo | 50 |  |  |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp35_cs_office | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_golden | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_palacio | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_vertigo | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_1_cs_office | 20 | 12.593 | 0.5586 | 0.6677 | 16.870 | 25.315 | 654.784 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_golden | 20 | 14.905 | 0.3799 | 0.6230 | 15.235 | 21.332 | 1142.936 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_palacio | 20 | 12.823 | 0.3570 | 0.7196 | 20.584 | 28.829 | 1095.749 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_vertigo | 20 | 11.647 | 0.4505 | 0.6785 | 19.523 | 27.623 | 956.651 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_cs_office | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_golden | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_palacio | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_vertigo | 20 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_cs_office | 20 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_golden | 20 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_palacio | 20 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_vertigo | 20 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp36_1_cs_office | 20 | 13.204 | 0.4811 | 0.7259 | 21.706 | 29.977 | 1855.592 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_golden | 20 | 13.300 | 0.4102 | 0.6512 | 18.424 | 27.034 | 1112.867 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_palacio | 20 | 13.014 | 0.4282 | 0.7040 | 20.486 | 29.211 | 1285.211 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_vertigo | 20 | 12.695 | 0.4485 | 0.6870 | 26.252 | 33.972 | 1383.501 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_cs_office | 20 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_gen_de_golden | 20 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_gen_de_palacio | 20 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_gen_de_vertigo | 20 |  |  |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp35_cs_office | 10 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_golden | 10 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_palacio | 10 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_de_vertigo | 10 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_1_cs_office | 10 | 11.602 | 0.5667 | 0.7173 | 17.355 | 26.503 | 763.896 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_golden | 10 | 14.461 | 0.3644 | 0.6274 | 14.456 | 19.487 | 1204.013 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_palacio | 10 | 12.377 | 0.3329 | 0.7080 | 19.888 | 26.938 | 1278.453 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_vertigo | 10 | 11.306 | 0.4552 | 0.6896 | 15.328 | 23.930 | 1255.499 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_cs_office | 10 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_golden | 10 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_palacio | 10 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen_de_vertigo | 10 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_cs_office | 10 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_golden | 10 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_palacio | 10 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_de_vertigo | 10 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp36_1_cs_office | 10 | 12.201 | 0.4505 | 0.7409 | 26.780 | 34.118 | 1842.580 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_golden | 10 | 12.821 | 0.3899 | 0.6691 | 16.990 | 25.150 | 1288.490 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_palacio | 10 | 13.242 | 0.4132 | 0.6978 | 23.892 | 31.541 | 1419.282 | 400 |
| Seen-10 retention | Continuous generation | exp36_1_de_vertigo | 10 | 12.297 | 0.4451 | 0.7063 | 22.483 | 30.050 | 1469.555 | 400 |
| Seen-10 retention | Continuous generation | exp35_gen_cs_office | 10 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_gen_de_golden | 10 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_gen_de_palacio | 10 |  |  |  |  |  |  | - |
| Seen-10 retention | Continuous generation | exp35_gen_de_vertigo | 10 |  |  |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp35 | 100 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35 | 50 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35 | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35 | 10 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp36_1 | 100 | 14.195 | 0.4437 | 0.6241 | 20.981 | 28.650 | 680.417 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1 | 50 | 13.459 | 0.4405 | 0.6567 | 20.004 | 27.902 | 831.237 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1 | 20 | 12.992 | 0.4365 | 0.6722 | 18.053 | 25.775 | 962.530 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp36_1 | 10 | 12.437 | 0.4298 | 0.6856 | 16.757 | 24.214 | 1125.465 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen | 100 | 13.943 | 0.4386 | 0.6306 | 22.033 | 29.725 | 667.223 | 400 |
| CrossMap-4 few-shot | Continuous generation | exp35_gen | 50 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen | 20 |  |  |  |  |  |  | - |
| CrossMap-4 few-shot | Continuous generation | exp35_gen | 10 |  |  |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|



# csgo benchmark v2 补充表格

## 定位

| Setting | Task | Experiment | Shot/map | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp33 | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp33 | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp33 | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp33 | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp33_loc | 100 | 175.491 | 11.717 | 3.968 | 76.271 |
| CrossMap-4 few-shot | Localization | exp33_loc | 50 | 195.684 | 13.497 | 3.995 | 84.102 |
| CrossMap-4 few-shot | Localization | exp33_loc | 20 | 199.455 | 14.185 | 3.662 | 82.913 |
| CrossMap-4 few-shot | Localization | exp33_loc | 10 | 216.629 | 14.493 | 3.790 | 86.198 |
| Seen-10 retention | Localization | exp33 | 100 |  |  |  |  |
| Seen-10 retention | Localization | exp33 | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp33 | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp33 | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp33_loc | 100 | 232.186 | 19.489 | 3.852 | 70.011 |
| Seen-10 retention | Localization | exp33_loc | 50 | 209.635 | 12.769 | 3.772 | 63.909 |
| Seen-10 retention | Localization | exp33_loc | 20 | 213.673 | 12.539 | 3.745 | 64.786 |
| Seen-10 retention | Localization | exp33_loc | 10 | 210.105 | 15.243 | 3.529 | 65.665 |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp34 | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp34 | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp34 | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp34 | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp34_loc | 100 | 155.607 | 11.146 | 3.516 | 73.515 |
| CrossMap-4 few-shot | Localization | exp34_loc | 50 | 187.740 | 14.105 | 3.992 | 80.578 |
| CrossMap-4 few-shot | Localization | exp34_loc | 20 | 193.382 | 13.611 | 4.235 | 82.967 |
| CrossMap-4 few-shot | Localization | exp34_loc | 10 | 214.016 | 13.774 | 4.187 | 86.504 |
| Seen-10 retention | Localization | exp34 | 100 |  |  |  |  |
| Seen-10 retention | Localization | exp34 | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp34 | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp34 | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp34_loc | 100 | 132.404 | 10.013 | 3.046 | 34.489 |
| Seen-10 retention | Localization | exp34_loc | 50 | 112.538 | 9.760 | 3.037 | 37.961 |
| Seen-10 retention | Localization | exp34_loc | 20 | 133.374 | 14.983 | 3.606 | 40.931 |
| Seen-10 retention | Localization | exp34_loc | 10 | 137.350 | 18.550 | 3.139 | 48.072 |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp36_cs_office | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_golden | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_palacio | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_vertigo | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_cs_office | 100 | 78.451 | 1.657 | 4.261 | 65.293 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_golden | 100 | 141.977 | 10.249 | 2.537 | 70.545 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_palacio | 100 | 198.323 | 12.683 | 3.086 | 62.587 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_vertigo | 100 | 144.532 | 15.911 | 3.411 | 70.993 |
| Seen-10 retention | Localization | exp36_cs_office | 100 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_golden | 100 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_palacio | 100 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_vertigo | 100 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_cs_office | 100 | 231.142 | 22.456 | 4.182 | 36.080 |
| Seen-10 retention | Localization | exp36_loc_de_golden | 100 | 119.687 | 13.227 | 3.040 | 43.915 |
| Seen-10 retention | Localization | exp36_loc_de_palacio | 100 | 116.260 | 10.093 | 2.983 | 36.384 |
| Seen-10 retention | Localization | exp36_loc_de_vertigo | 100 | 145.808 | 7.935 | 3.287 | 34.537 |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp36_cs_office | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_golden | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_palacio | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_vertigo | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_cs_office | 50 | 108.359 | 2.364 | 4.890 | 78.328 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_golden | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_palacio | 50 | 236.262 | 16.242 | 3.250 | 74.013 |
| CrossMap-4 few-shot | Localization | exp36_loc_de_vertigo | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp36_cs_office | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_golden | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_palacio | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_vertigo | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_cs_office | 50 | 227.768 | 27.347 | 4.594 | 42.501 |
| Seen-10 retention | Localization | exp36_loc_de_golden | 50 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_palacio | 50 | 182.345 | 14.171 | 3.322 | 33.873 |
| Seen-10 retention | Localization | exp36_loc_de_vertigo | 50 |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp36_cs_office | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_golden | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_palacio | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_vertigo | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_cs_office | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_golden | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_palacio | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_vertigo | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_cs_office | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_golden | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_palacio | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_vertigo | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_palacio | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_golden | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_cs_office | 20 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_vertigo | 20 |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp36_cs_office | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_golden | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_palacio | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_de_vertigo | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_cs_office | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_golden | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_palacio | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc_de_vertigo | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_cs_office | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_golden | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_palacio | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_de_vertigo | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_vertigo | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_cs_office | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_golden | 10 |  |  |  |  |
| Seen-10 retention | Localization | exp36_loc_de_palacio | 10 |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp36 | 100 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36 | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36 | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36 | 10 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc | 100 | 140.821 | 10.125 | 3.324 | 67.354 |
| CrossMap-4 few-shot | Localization | exp36_loc | 50 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc | 20 |  |  |  |  |
| CrossMap-4 few-shot | Localization | exp36_loc | 10 |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Localization | exp36_1_cs_office | 50 | 122.239 | 2.100 | 4.750 | 81.182 |
| Seen-10 retention | Localization | exp36_1_cs_office | 50 | 287.580 | 18.498 | 8.212 | 60.659 |
| CrossMap-4 few-shot | Localization | exp36_1_de_golden | 50 | 175.451 | 12.672 | 2.785 | 80.600 |
| Seen-10 retention | Localization | exp36_1_de_golden | 50 | 198.294 | 12.191 | 3.734 | 68.007 |
| CrossMap-4 few-shot | Localization | exp36_1_de_palacio | 50 | 258.732 | 14.284 | 3.558 | 73.738 |
| Seen-10 retention | Localization | exp36_1_de_palacio | 50 | 231.973 | 16.715 | 3.883 | 61.935 |
| CrossMap-4 few-shot | Localization | exp36_1_de_vertigo | 50 | 167.365 | 17.538 | 3.446 | 81.584 |
| Seen-10 retention | Localization | exp36_1_de_vertigo | 50 | 229.513 | 11.275 | 5.039 | 61.304 |
| CrossMap-4 few-shot | Localization | exp36_1 | 50 | 180.947 | 11.649 | 3.635 | 79.276 |


## 离散生成

| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp33 | 100 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp33 | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp33 | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp33 | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp33_gen | 100 | 13.520 | 0.4413 | 0.6475 | 0.4867 | 35.407 |
| CrossMap-4 few-shot | Discrete generation | exp33_gen | 50 | 13.462 | 0.4623 | 0.6590 | 0.4785 | 48.555 |
| CrossMap-4 few-shot | Discrete generation | exp33_gen | 20 | 13.267 | 0.4509 | 0.6654 | 0.4781 | 53.425 |
| CrossMap-4 few-shot | Discrete generation | exp33_gen | 10 | 12.795 | 0.4483 | 0.6815 | 0.4676 | 67.969 |
| Seen-10 retention | Discrete generation | exp33 | 100 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp33 | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp33 | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp33 | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp33_gen | 100 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp33_gen | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp33_gen | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp33_gen | 10 |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp34 | 100 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp34 | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp34 | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp34 | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp34_gen | 100 | 13.138 | 0.4309 | 0.6613 | 0.4826 | 38.850 |
| CrossMap-4 few-shot | Discrete generation | exp34_gen | 50 | 13.264 | 0.4419 | 0.6599 | 0.4771 | 43.035 |
| CrossMap-4 few-shot | Discrete generation | exp34_gen | 20 | 13.096 | 0.4362 | 0.6652 | 0.4723 | 44.739 |
| CrossMap-4 few-shot | Discrete generation | exp34_gen | 10 | 12.784 | 0.4283 | 0.6756 | 0.4642 | 52.337 |
| Seen-10 retention | Discrete generation | exp34 | 100 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp34 | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp34 | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp34 | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp34_gen | 100 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp34_gen | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp34_gen | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp34_gen | 10 |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp36_cs_office | 100 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_golden | 100 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_palacio | 100 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_vertigo | 100 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_cs_office | 100 | 14.340 | 0.5913 | 0.6085 | 0.4673 | 34.787 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_golden | 100 | 15.160 | 0.3727 | 0.6012 | 0.5336 | 36.074 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_palacio | 100 | 13.205 | 0.3696 | 0.6743 | 0.5222 | 33.248 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_vertigo | 100 | 12.649 | 0.4548 | 0.6477 | 0.4578 | 35.587 |
| Seen-10 retention | Discrete generation | exp36_cs_office | 100 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_golden | 100 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_palacio | 100 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_vertigo | 100 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_cs_office | 100 | 12.806 | 0.4319 | 0.6903 | 0.4698 | 94.478 |
| Seen-10 retention | Discrete generation | exp36_gen_de_golden | 100 | 13.070 | 0.3718 | 0.6554 | 0.5010 | 69.402 |
| Seen-10 retention | Discrete generation | exp36_gen_de_palacio | 100 | 12.724 | 0.3869 | 0.6746 | 0.4858 | 59.855 |
| Seen-10 retention | Discrete generation | exp36_gen_de_vertigo | 100 | 12.095 | 0.4035 | 0.6825 | 0.4676 | 78.333 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp36_cs_office | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_golden | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_palacio | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_vertigo | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_cs_office | 50 | 13.584 | 0.5890 | 0.6393 | 0.4501 | 38.422 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_golden | 50 | 14.922 | 0.3689 | 0.6108 | 0.5240 | 36.104 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_palacio | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_vertigo | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_cs_office | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_golden | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_palacio | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_vertigo | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_cs_office | 50 | 12.823 | 0.4382 | 0.6977 | 0.4656 | 103.438 |
| Seen-10 retention | Discrete generation | exp36_gen_de_golden | 50 | 13.079 | 0.3742 | 0.6528 | 0.5034 | 64.711 |
| Seen-10 retention | Discrete generation | exp36_gen_de_palacio | 50 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_vertigo | 50 |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp36_cs_office | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_golden | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_palacio | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_vertigo | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_cs_office | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_golden | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_palacio | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_vertigo | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_cs_office | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_golden | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_palacio | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_vertigo | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_cs_office | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_golden | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_palacio | 20 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_vertigo | 20 |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp36_cs_office | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_golden | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_palacio | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_de_vertigo | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_cs_office | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_golden | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_palacio | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen_de_vertigo | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_cs_office | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_golden | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_palacio | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_de_vertigo | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_cs_office | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_golden | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_palacio | 10 |  |  |  |  |  |
| Seen-10 retention | Discrete generation | exp36_gen_de_vertigo | 10 |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp36 | 100 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36 | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36 | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36 | 10 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen | 100 | 13.839 | 0.4471 | 0.6329 | 0.4952 | 34.924 |
| CrossMap-4 few-shot | Discrete generation | exp36_gen | 50 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen | 20 |  |  |  |  |  |
| CrossMap-4 few-shot | Discrete generation | exp36_gen | 10 |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Discrete generation | exp36_1_cs_office | 50 | 14.086 | 0.6068 | 0.6216 | 0.4516 | 46.612 |
| Seen-10 retention | Discrete generation | exp36_1_cs_office | 50 | 13.295 | 0.4653 | 0.7033 | 0.4790 | 126.942 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_golden | 50 | 15.211 | 0.3760 | 0.6136 | 0.5346 | 51.181 |
| Seen-10 retention | Discrete generation | exp36_1_de_golden | 50 | 13.518 | 0.3976 | 0.6409 | 0.5195 | 75.333 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_palacio | 50 | 13.312 | 0.3794 | 0.6892 | 0.5218 | 52.357 |
| Seen-10 retention | Discrete generation | exp36_1_de_palacio | 50 | 13.613 | 0.4332 | 0.6758 | 0.5001 | 92.391 |
| CrossMap-4 few-shot | Discrete generation | exp36_1_de_vertigo | 50 | 12.377 | 0.4614 | 0.6544 | 0.4599 | 50.240 |
| Seen-10 retention | Discrete generation | exp36_1_de_vertigo | 50 | 13.036 | 0.4476 | 0.6646 | 0.4778 | 86.895 |
| CrossMap-4 few-shot | Discrete generation | exp36_1 | 50 | 13.746 | 0.4559 | 0.6447 | 0.4920 | 50.098 |



## 连续生成

| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | FVD↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp33 | 100 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp33 | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp33 | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp33 | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp33_gen | 100 | 13.233 | 0.4230 | 0.6555 | 31.991 | 38.899 | 853.537 |
| CrossMap-4 few-shot | Continuous generation | exp33_gen | 50 | 13.061 | 0.4458 | 0.6731 | 22.406 | 30.492 | 839.893 |
| CrossMap-4 few-shot | Continuous generation | exp33_gen | 20 | 12.774 | 0.4315 | 0.6763 | 21.927 | 29.860 | 896.970 |
| CrossMap-4 few-shot | Continuous generation | exp33_gen | 10 | 12.532 | 0.4301 | 0.6842 | 22.488 | 30.143 | 955.571 |
| Seen-10 retention | Continuous generation | exp33 | 100 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp33 | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp33 | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp33 | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp33_gen | 100 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp33_gen | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp33_gen | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp33_gen | 10 |  |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp34 | 100 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp34 | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp34 | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp34 | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp34_gen | 100 | 12.896 | 0.4125 | 0.6691 | 38.268 | 44.353 | 934.424 |
| CrossMap-4 few-shot | Continuous generation | exp34_gen | 50 | 12.935 | 0.4236 | 0.6671 | 31.576 | 38.439 | 975.876 |
| CrossMap-4 few-shot | Continuous generation | exp34_gen | 20 | 12.805 | 0.4182 | 0.6700 | 31.074 | 37.991 | 895.785 |
| CrossMap-4 few-shot | Continuous generation | exp34_gen | 10 | 12.521 | 0.4106 | 0.6796 | 31.051 | 37.968 | 911.294 |
| Seen-10 retention | Continuous generation | exp34 | 100 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp34 | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp34 | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp34 | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp34_gen | 100 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp34_gen | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp34_gen | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp34_gen | 10 |  |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp36_cs_office | 100 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_golden | 100 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_palacio | 100 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_vertigo | 100 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_cs_office | 100 | 14.284 | 0.5629 | 0.6254 | 26.457 | 33.674 | 570.938 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_golden | 100 | 15.211 | 0.3697 | 0.5928 | 25.736 | 31.647 | 870.651 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_palacio | 100 | 12.949 | 0.3514 | 0.6807 | 33.594 | 40.721 | 846.198 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_vertigo | 100 | 12.527 | 0.4449 | 0.6416 | 36.199 | 43.518 | 729.287 |
| Seen-10 retention | Continuous generation | exp36_cs_office | 100 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_golden | 100 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_palacio | 100 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_vertigo | 100 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_cs_office | 100 | 12.830 | 0.4315 | 0.6910 | 33.353 | 39.939 | 1518.554 |
| Seen-10 retention | Continuous generation | exp36_gen_de_golden | 100 | 13.114 | 0.3697 | 0.6590 | 29.135 | 36.264 | 1004.852 |
| Seen-10 retention | Continuous generation | exp36_gen_de_palacio | 100 | 12.951 | 0.3915 | 0.6693 | 34.706 | 41.255 | 1011.143 |
| Seen-10 retention | Continuous generation | exp36_gen_de_vertigo | 100 | 12.217 | 0.4036 | 0.6819 | 40.245 | 46.283 | 1258.807 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp36_cs_office | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_golden | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_palacio | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_vertigo | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_cs_office | 50 | 12.967 | 0.5558 | 0.6730 | 29.120 | 36.126 | 610.194 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_golden | 50 | 15.159 | 0.3696 | 0.5983 | 23.993 | 30.215 | 855.723 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_palacio | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_vertigo | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_cs_office | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_golden | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_palacio | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_vertigo | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_cs_office | 50 | 12.789 | 0.4353 | 0.6978 | 33.573 | 40.052 | 1612.398 |
| Seen-10 retention | Continuous generation | exp36_gen_de_golden | 50 | 13.061 | 0.3690 | 0.6573 | 28.247 | 35.488 | 1006.422 |
| Seen-10 retention | Continuous generation | exp36_gen_de_palacio | 50 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_vertigo | 50 |  |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp36_cs_office | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_golden | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_palacio | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_vertigo | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_cs_office | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_golden | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_palacio | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_vertigo | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_cs_office | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_golden | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_palacio | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_vertigo | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_cs_office | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_golden | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_palacio | 20 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_vertigo | 20 |  |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp36_cs_office | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_golden | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_palacio | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_de_vertigo | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_cs_office | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_golden | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_palacio | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen_de_vertigo | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_cs_office | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_golden | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_palacio | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_de_vertigo | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_cs_office | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_golden | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_palacio | 10 |  |  |  |  |  |  |
| Seen-10 retention | Continuous generation | exp36_gen_de_vertigo | 10 |  |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp36 | 100 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36 | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36 | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36 | 10 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen | 100 | 13.743 | 0.4322 | 0.6351 | 30.497 | 37.390 | 754.268 |
| CrossMap-4 few-shot | Continuous generation | exp36_gen | 50 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen | 20 |  |  |  |  |  |  |
| CrossMap-4 few-shot | Continuous generation | exp36_gen | 10 |  |  |  |  |  |  |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CrossMap-4 few-shot | Continuous generation | exp36_1_cs_office | 50 | 13.434 | 0.5723 | 0.6559 | 20.370 | 28.521 | 615.408 |
| Seen-10 retention | Continuous generation | exp36_1_cs_office | 50 | 13.329 | 0.4650 | 0.7057 | 24.584 | 32.383 | 1729.202 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_golden | 50 | 15.350 | 0.3772 | 0.6054 | 17.216 | 23.258 | 864.206 |
| Seen-10 retention | Continuous generation | exp36_1_de_golden | 50 | 13.578 | 0.3972 | 0.6418 | 21.846 | 29.934 | 1020.407 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_palacio | 50 | 12.840 | 0.3641 | 0.7152 | 19.197 | 28.217 | 977.899 |
| Seen-10 retention | Continuous generation | exp36_1_de_palacio | 50 | 13.918 | 0.4397 | 0.6657 | 22.906 | 31.142 | 1185.522 |
| CrossMap-4 few-shot | Continuous generation | exp36_1_de_vertigo | 50 | 12.213 | 0.4485 | 0.6503 | 23.231 | 31.611 | 867.434 |
| Seen-10 retention | Continuous generation | exp36_1_de_vertigo | 50 | 13.272 | 0.4495 | 0.6603 | 27.879 | 35.415 | 1282.012 |
| CrossMap-4 few-shot | Continuous generation | exp36_1 | 50 | 13.459 | 0.4405 | 0.6567 | 20.004 | 27.902 | 831.237 |

# exp37 Seen-10 两阶段实验（2026-11-04 同步结果）

协议见 [Seen-10两阶段实验](CSGO_BENCHMARK_V2_EXP37_SEEN10.md)，命令见 [record.md](record.md#exp37-seen10-2026-10-30)。本次读取已同步的训练状态、推理清单和指标JSON，不启动实验。已核实四组训练完成及相应validation／test指标；两个both候选本地尚无产物，不能据此判断远端是否在运行。结果来自完整十图、minimal资产、seed42、等地图宏平均；定位与离散test各20000条，连续test为200×64=12800帧。三列参数量本次不做新审计，保留待审计。

| 实验版本 | 来源与训练身份 | 训练 | validation | test / metrics | 表格 checkpoint |
|---|---|---|---|---|---|
| `exp37` | 原始UniLIP＋π0.5；joint J8，αloc=8，无辅助 | ✅ 19550/19550，50 epochs | ✅ @8000、19550；定位／离散 | ✅ 定位／离散／连续 | Stage 1 final 19550，仅参照 |
| `exp37_s2_gen` | `exp32_gen@19500`仅模型；生成单任务 | ✅ local6000/6000 | ✅ @4000、6000；离散 | ✅ 离散／连续 | local6000 |
| `exp37_s2_loc` | `exp32_loc@19500`仅模型；定位单任务 | ✅ local6000/6000 | ✅ @4000、6000；定位 | ✅ 定位 | local6000 |
| `exp37_s2` | `exp37@19550`仅模型；joint无辅助 | ✅ local6000/6000 | ✅ @4000、6000；定位／离散 | ✅ 定位／离散／连续 | local6000 |
| `exp37_s2_both` | `exp37@19550`；joint P4＋U2 | 未同步，远端状态待确认 | 未同步 | 未同步 | local6000（计划） |
| `exp37_s2_both_b1` | `exp37@19550`；joint P3＋U2 | 未同步，远端状态待确认 | 未同步 | 未同步 | local6000（计划） |

Stage2首次只加载来源模型，重建optimizer／scheduler／local step／RNG；其预算均为local6000、有效base batch128。两组both的配方只在perception attention=false／true上不同。validation只评定位和离散生成，连续生成按协议只在test评测，不应标成缺失的validation任务。各validation节点的关键指标与来源见 [专题结果表](CSGO_BENCHMARK_V2_EXP37_SEEN10.md#数据评测和结果记录)。

## exp37 定位（Seen-10；最终test macro）

| Experiment | Checkpoint | Total params | Active params | Trainable params | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 来源 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exp37` | Stage 1 final 19550 | 待审计 | 待审计 | 待审计 | 49.9381 | 2.9983 | 2.6749 | 26.8646 | [JSON](outputs_loc/benchmark_v2/exp37/checkpoint-19550/test/seed_42/benchmark_csgo_v2_loc.json) |
| `exp37_s2_loc` | local 6000 | 待审计 | 待审计 | 待审计 | 46.3528 | 2.8725 | 2.5183 | 25.1111 | [JSON](outputs_loc/benchmark_v2/exp37_s2_loc/checkpoint-6000/test/seed_42/benchmark_csgo_v2_loc.json) |
| `exp37_s2` | local 6000 | 待审计 | 待审计 | 待审计 | 47.1936 | 2.8386 | 2.6494 | 26.3374 | [JSON](outputs_loc/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/benchmark_csgo_v2_loc.json) |
| `exp37_s2_both` | local 6000 | 待审计 | 待审计 | 待审计 | 未同步 | 未同步 | 未同步 | 未同步 | — |
| `exp37_s2_both_b1` | local 6000 | 待审计 | 待审计 | 待审计 | 未同步 | 未同步 | 未同步 | 未同步 | — |

## exp37 离散生成（Seen-10；最终test macro）

| Experiment | Checkpoint | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 来源 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exp37` | Stage 1 final 19550 | 待审计 | 待审计 | 待审计 | 13.6163 | 0.6272 | 0.5078 | 36.3942 | [JSON](outputs_eval/benchmark_v2/exp37/checkpoint-19550/test/seed_42/discrete/summary.json) |
| `exp37_s2_gen` | local 6000 | 待审计 | 待审计 | 待审计 | 14.1668 | 0.6029 | 0.5217 | 34.0804 | [JSON](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/discrete/summary.json) |
| `exp37_s2` | local 6000 | 待审计 | 待审计 | 待审计 | 13.7481 | 0.6229 | 0.5099 | 35.8037 | [JSON](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/discrete/summary.json) |
| `exp37_s2_both` | local 6000 | 待审计 | 待审计 | 待审计 | 未同步 | 未同步 | 未同步 | 未同步 | — |
| `exp37_s2_both_b1` | local 6000 | 待审计 | 待审计 | 待审计 | 未同步 | 未同步 | 未同步 | 未同步 | — |

## exp37 连续生成（Seen-10；最终test macro）

| Experiment | Checkpoint | Total params | Active params | Trainable params | PSNR↑ | LPIPS↓ | T-Warp↓ | T-Diff↓ | FVD↓ | 来源 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exp37` | Stage 1 final 19550 | 待审计 | 待审计 | 待审计 | 13.8818 | 0.6171 | 37.9583 | 43.9776 | 943.6128 | [JSON](outputs_eval/benchmark_v2/exp37/checkpoint-19550/test/seed_42/continuous/summary.json) |
| `exp37_s2_gen` | local 6000 | 待审计 | 待审计 | 待审计 | 14.6232 | 0.5814 | 34.5009 | 40.6948 | 843.4360 | [JSON](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/continuous/summary.json) |
| `exp37_s2` | local 6000 | 待审计 | 待审计 | 待审计 | 14.0911 | 0.6077 | 37.1104 | 43.1822 | 922.5427 | [JSON](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/continuous/summary.json) |
| `exp37_s2_both` | local 6000 | 待审计 | 待审计 | 待审计 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | — |
| `exp37_s2_both_b1` | local 6000 | 待审计 | 待审计 | 待审计 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | — |

## 2026-11-04：exp37 阶段性结果与 Seen-10 其他模型对照

以下对照使用固定最终 checkpoint 的 test 等地图宏平均；外部模型沿用 [Table 1](csgo_benchmark_v2_table1_seen10.md) 已登记数值，本次未重新评测外部项目。validation 单独记录，不与 test 的 FID 混比（每图分别500和2000张）。

- **续训有效，但联合尚未整体超过单任务。** `exp37_s2` 相比 `exp37`，XY从49.9381降至47.1936（−5.50%），离散PSNR从13.6163升至13.7481，连续FVD从943.6128降至922.5427；不过对齐Stage2预算后，定位单任务XY=46.3528，生成单任务PSNR=14.1668、FID=34.0804、FVD=843.4360。联合的Z稍好（2.8386 vs 2.8725），XY／Pitch／Yaw及所列生成指标均落后。当前不支持“联合全面优于单任务”的结论，也不能将改善归因于尚无结果的辅助损失。
- **定位已恢复到接近单任务的水平，外部强基线仍有优势。** 联合XY=47.1936明显好于旧 `exp32` 的123.194和 `exp32_1` 的63.0377；但低于旧模型的误差并不单独证明两阶段设计，因为主损失权重、学习率与训练预算也变化。与外部基线相比，XY优于X-VLA的65.969和π0.5的53.542，仍落后RDT的43.251与OpenVLA-OFT的36.956。各角度误差须分别报告，不能用XY代替整个5D定位能力。
- **生成单任务的FVD有竞争力，图像质量尚有差距。** `exp37_s2_gen` FVD=843.4360，在当前Table 1已登记模型与本次新结果中数值最低，接近Puffin的844.2427；这不代表所有生成指标领先。其离散PSNR=14.1668、LPIPS=0.6029弱于Show-o2（14.539／0.591）和Puffin（14.3186／0.5924）；FID=34.0804也高于ControlAR的30.2869。联合组生成指标整体比对应单任务差，仍是当前主要短板。
- **下一步先补齐证据。** 同步两组both（P4＋U2、P3＋U2）的validation@4000／6000和test@6000，按同一基座、同一预算比较辅助收益；最终辅助方案依据Seen validation确定。Unseen已完成的只是gen／loc／joint的zero-shot基线，后续独立400步适配及Seen-retention仍待同步，不能以zero-shot结果宣称few-shot收益，也不能用Unseen query挑选配方。本文两阶段结果不声称与旧单阶段外部模型等FLOPs。

<a id="exp32-3maps-b2-s3-j8-20261030"></a>

<a id="b2-ablation-summary-20261030"></a>

# exp32_3maps B2_S3_J8 消融补充（2026-10-30 设计；2026-11-01 结果同步）

本节属于 **ablation 3 maps**，不进入 Seen-10 主表；B2 是第 38 个计划运行名。2026-10-30 时仅有配置和占位；2026-11-01 已同步完整实测结果，历史 24／33／35／37 组规划与早期 16 模型 test 排名保留当时口径。[Seen-3 详细配方与结果](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#b2-stage2-j8-20261030)、[Stage 2 设计](CSGO_BENCHMARK_V2_EXP32_3MAPS_STAGE2.md#b2-stage2-j8-20261030)、[手动命令](record.md#exp32_3maps-b2_s3_j8-2026-10-30-手动命令)分别记录方案、比较口径及运行方式。

## ablation 3 maps：B2 配置与进度

| 实验版本 | 来源／配方 | 训练 | seen_validation local1200 | seen_validation local1800 | test local1800 |
|---|---|---|---|---|---|
| `exp32_3maps_B2_S3_J8` | J8@5850 仅模型权重；P4＋U2，perception attention=false；αloc=8；三图 paired joint | ✅ local1800/1800，双卡 micro4/GAS16，名义有效 batch128；[状态](outputs/csgo_1b/exp32_3maps_B2_S3_J8/trainer_state.json) | ✅ 定位＋离散 | ✅ 定位＋离散 | ✅ 定位＋离散＋连续 |

## ablation 3 maps：B2 实测指标（2026-11-01）

| Split／checkpoint | 定位 XY／Z／Pitch／Yaw ↓ | 离散 PSNR↑／SSIM↑／LPIPS↓／Boundary F1↑／FID↓ | 连续 PSNR↑／SSIM↑／LPIPS↓／Boundary F1↑／TWE↓／TDE↓／OFE↓／FVD↓ | 状态 |
|---|---|---|---|---|
| seen_validation／local1200 | 47.515／2.809／2.520／24.489 | 14.387／0.418／0.619／0.519／63.723 | N/A | [定位](outputs_loc/benchmark_v2/exp32_3maps_B2_S3_J8/checkpoint-1200/validation/seed_42/loc_metrics.json)／[离散](outputs_eval/benchmark_v2/exp32_3maps_B2_S3_J8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| seen_validation／local1800 | 44.504／2.649／2.458／24.232 | 14.405／0.418／0.614／0.522／62.376 | N/A | [定位](outputs_loc/benchmark_v2/exp32_3maps_B2_S3_J8/checkpoint-1800/validation/seed_42/loc_metrics.json)／[离散](outputs_eval/benchmark_v2/exp32_3maps_B2_S3_J8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| seen_discrete_test＋seen_continuous／local1800 | 53.342／2.902／2.725／30.137 | 14.100／0.400／0.631／0.518／36.172 | 14.956／0.431／0.593／0.537／33.787／38.982／16.201／896.644 | [定位](outputs_loc/benchmark_v2/exp32_3maps_B2_S3_J8/checkpoint-1800/test/seed_42/loc_metrics.json)／[离散](outputs_eval/benchmark_v2/exp32_3maps_B2_S3_J8/checkpoint-1800/test/seed_42/discrete/summary.json)／[连续](outputs_eval/benchmark_v2/exp32_3maps_B2_S3_J8/checkpoint-1800/test/seed_42/continuous/summary.json) |

同 local1200／1800 验证节点比较 S0_J8、P4、U2、B2 的无辅助／单项／联合四格；B1 与 B2 只比较 perception attention=true／false。test 仅用于报告；B2 的 2026-11-01 实测行已计入主文档 21 模型汇总。Seen-10 `exp37_s2_both` 与 B2 同为 P4＋U2 形式，但使用十图、ramp400、local6000，结果必须分开呈现。

<a id="u4-b1-results-20261101"></a>

# exp32_3maps U4／B1 Stage 2 结果补充（2026-11-01）

U4、B1 均从 J8@5850 仅模型权重初始化，完成 local1800；本地计划为单卡 microbatch=4、GAS=32、seed=42。两组均有 local1200／1800 定位与离散生成 validation，以及 local1800 定位、离散、连续 test。源 JSON 的 `metrics_macro_map` 为三图等权宏平均；validation 每图500条，test 定位／离散每图2,000条、连续每图20×64帧。当前本地 U4 模型 local1200／1800、B1 模型 local600／1200／1800 存在；状态、计划、resolved 配方已核对，未核实完整恢复材料。详见[九组主进度与结果](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#nine-ablation-results-20261026)及[test 总表](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#test-overall-20261101)。

| 实验版本 | 训练 | validation local1200／1800 | test local1800 | 本地证据 |
|---|---|---|---|---|
| `exp32_3maps_B1_S3_J8` | ✅ local1800/1800 | ✅ 定位＋离散生成 | ✅ 定位＋离散＋连续生成 | [Trainer 状态](outputs/csgo_1b/exp32_3maps_B1_S3_J8/trainer_state.json)、[计划](outputs/csgo_1b/exp32_3maps_B1_S3_J8/experiment_plan.json)、[配方](outputs/csgo_1b/exp32_3maps_B1_S3_J8/resolved_config.yaml) |
| `exp32_3maps_U4_S2_J8` | ✅ local1800/1800 | ✅ 定位＋离散生成 | ✅ 定位＋离散＋连续生成 | [Trainer 状态](outputs/csgo_1b/exp32_3maps_U4_S2_J8/trainer_state.json)、[计划](outputs/csgo_1b/exp32_3maps_U4_S2_J8/experiment_plan.json)、[配方](outputs/csgo_1b/exp32_3maps_U4_S2_J8/resolved_config.yaml) |

## 定位（validation／test）

| 实验 | split | local step | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 源 JSON |
|---|---|---:|---:|---:|---:|---:|---|
| `exp32_3maps_B1_S3_J8` | seen_validation | 1200 | 45.455 | 2.740 | 2.363 | 24.659 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_B1_S3_J8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| `exp32_3maps_B1_S3_J8` | seen_validation | 1800 | 44.662 | 2.680 | 2.589 | 24.307 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_B1_S3_J8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| `exp32_3maps_B1_S3_J8` | seen_discrete_test | 1800 | 54.101 | 2.965 | 2.801 | 30.245 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_B1_S3_J8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| `exp32_3maps_U4_S2_J8` | seen_validation | 1200 | 45.642 | 2.691 | 2.505 | 24.458 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U4_S2_J8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| `exp32_3maps_U4_S2_J8` | seen_validation | 1800 | 44.312 | 2.622 | 2.371 | 24.060 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U4_S2_J8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| `exp32_3maps_U4_S2_J8` | seen_discrete_test | 1800 | 53.223 | 2.894 | 2.621 | 30.112 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U4_S2_J8/checkpoint-1800/test/seed_42/loc_metrics.json) |

## 离散生成（validation／test）

| 实验 | split | local step | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | 源 JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `exp32_3maps_B1_S3_J8` | seen_validation | 1200 | 14.308 | 0.413 | 0.618 | 0.520 | 63.382 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_B1_S3_J8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| `exp32_3maps_B1_S3_J8` | seen_validation | 1800 | 14.394 | 0.415 | 0.615 | 0.523 | 63.806 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_B1_S3_J8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| `exp32_3maps_B1_S3_J8` | seen_discrete_test | 1800 | 14.060 | 0.395 | 0.633 | 0.517 | 37.289 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_B1_S3_J8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| `exp32_3maps_U4_S2_J8` | seen_validation | 1200 | 14.228 | 0.411 | 0.620 | 0.520 | 63.410 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U4_S2_J8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| `exp32_3maps_U4_S2_J8` | seen_validation | 1800 | 14.390 | 0.416 | 0.615 | 0.522 | 62.895 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U4_S2_J8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| `exp32_3maps_U4_S2_J8` | seen_discrete_test | 1800 | 14.052 | 0.396 | 0.633 | 0.517 | 37.403 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U4_S2_J8/checkpoint-1800/test/seed_42/discrete/summary.json) |

## 连续生成（test）

| 实验 | split | local step | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | TWE↓ | TDE↓ | OFE↓ | FVD↓ | 源 JSON |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `exp32_3maps_B1_S3_J8` | seen_continuous | 1800 | 15.074 | 0.433 | 0.585 | 0.541 | 33.046 | 38.262 | 16.089 | 916.709 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_B1_S3_J8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| `exp32_3maps_U4_S2_J8` | seen_continuous | 1800 | 14.990 | 0.431 | 0.593 | 0.535 | 33.720 | 38.901 | 16.162 | 874.366 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U4_S2_J8/checkpoint-1800/test/seed_42/continuous/summary.json) |

U4／B1 与独立 B2 的直接对照、test 取舍及后续优先级见[2026-11-01 消融分析](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#u4-both-analysis-20261101)。

<a id="exp37-unseen4-results-20261103"></a>

# exp37 Unseen-4 每图独立 few-shot（2026-11-04 同步结果）

本节同步新实验进度与结果，不改变历史表格顺序和已有数值。协议见 [exp37 Unseen-4](CSGO_BENCHMARK_V2_EXP37_UNSEEN4.md)，手动命令见 [record.md](record.md#exp37-unseen4-fewshot-20261103)。五个Seen Stage2@6000来源、四图、100/50/20/10-shot，共80次独立400步适配；若预先只选一个both则实际64次。已同步gen／loc／joint三组zero-shot；尚未找到本系列80个独立400步适配点的训练、query或retention产物。下表“未同步”仅描述本机证据，不表示指标为零，也不推断远端尚未启动。本次未运行任何训练、推理或评测。

## zero-shot 实验进度

| Family | 未适配Seen来源 | 推理 | Metric | 正式checkpoint |
|---|---|---|---|---|
| exp37_fs_gen | exp37_s2_gen | ✅ 离散／连续 | ✅ 四图完整宏平均 | Seen local6000 |
| exp37_fs_loc | exp37_s2_loc | ✅ 定位 | ✅ 四图完整宏平均 | Seen local6000 |
| exp37_fs_joint | exp37_s2 | ✅ 定位／离散／连续 | ✅ 四图完整宏平均 | Seen local6000 |
| exp37_fs_both | exp37_s2_both | 未同步（joint） | 未同步 | Seen local6000 |
| exp37_fs_both_b1 | exp37_s2_both_b1 | 未同步（joint） | 未同步 | Seen local6000 |

## few-shot 实验进度

各shot单元格依次记录训练／目标地图query／全Seen10 retention；每个单元格是一次独立适配，而不是从前一个shot继续训练。新实验开启sampler epoch转发；joint实际每步100/48/20/8个base样本，详见协议。

| 实验版本 | 100-shot：训练／Q／R | 50-shot：训练／Q／R | 20-shot：训练／Q／R | 10-shot：训练／Q／R | checkpoint／validation |
|---|---|---|---|---|---|
| exp37_fs_gen_cs_office | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_gen_de_golden | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_gen_de_palacio | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_gen_de_vertigo | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_loc_cs_office | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_loc_de_golden | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_loc_de_palacio | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_loc_de_vertigo | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_joint_cs_office | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_joint_de_golden | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_joint_de_palacio | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_joint_de_vertigo | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_both_cs_office | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_both_de_golden | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_both_de_palacio | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_both_de_vertigo | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_both_b1_cs_office | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_both_b1_de_golden | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_both_b1_de_palacio | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |
| exp37_fs_both_b1_de_vertigo | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | 未同步／未同步／未同步 | final400；不做Unseen中途选点 |

zero-shot使用同一个Seen checkpoint覆盖四图；few-shot macro使用四个目标图专用checkpoint，以 `map-models` 聚合，保留map→checkpoint对应关系。每个适配模型的Seen-retention单独记录，不将四个模型拼为单模型retention。下面填入已完成zero-shot的四图宏平均；few-shot保留未同步标记。逐图zero-shot指标与完整来源见 [Unseen专题文档](CSGO_BENCHMARK_V2_EXP37_UNSEEN4.md)。

生成输出位于 `outputs_eval/benchmark_v2/<family>_<map>/shot_<N>/seed_0/`，定位位于同结构的 `outputs_loc/benchmark_v2/`；retention为本次运行的独立子目录。zero-shot单独使用 `<family>/zero_shot/seed_42/`。只在严格核对来源、step、seed、shot、manifest、后端及覆盖后填数值。

## exp37 Unseen-4 定位（四图宏平均）

| Family | Shot／设置 | Checkpoint | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ |
|---|---|---|---:|---:|---:|---:|
| exp37_fs_loc | 0-shot | Seen local6000 | 339.9282 | 19.8735 | 7.6095 | 91.5718 |
| exp37_fs_loc | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_loc | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_loc | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_loc | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 0-shot | Seen local6000 | 344.4904 | 18.7200 | 7.8164 | 92.0992 |
| exp37_fs_joint | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 0-shot | Seen local6000 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 0-shot | Seen local6000 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |

## exp37 Unseen-4 离散生成（四图宏平均）

| Family | Shot／设置 | Checkpoint | PSNR↑ | LPIPS↓ | Boundary_F1↑ | FID↓ |
|---|---|---|---:|---:|---:|---:|
| exp37_fs_gen | 0-shot | Seen local6000 | 11.3790 | 0.7169 | 0.4578 | 77.4963 |
| exp37_fs_gen | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_gen | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_gen | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_gen | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 0-shot | Seen local6000 | 11.0413 | 0.7288 | 0.4605 | 103.5831 |
| exp37_fs_joint | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 0-shot | Seen local6000 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 0-shot | Seen local6000 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 |

## exp37 Unseen-4 连续生成（四图宏平均）

| Family | Shot／设置 | Checkpoint | PSNR↑ | LPIPS↓ | T-Warp↓ | T-Diff↓ | FVD↓ |
|---|---|---|---:|---:|---:|---:|---:|
| exp37_fs_gen | 0-shot | Seen local6000 | 11.3554 | 0.7167 | 44.5674 | 50.1917 | 1283.9551 |
| exp37_fs_gen | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_gen | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_gen | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_gen | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 0-shot | Seen local6000 | 11.1061 | 0.7282 | 42.7217 | 48.6215 | 1368.5584 |
| exp37_fs_joint | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_joint | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 0-shot | Seen local6000 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 0-shot | Seen local6000 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 100-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 50-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 20-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| exp37_fs_both_b1 | 10-shot | 四个map-specific local400 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |

历史待补项与本节新增实验分开：exp35 joint四shot共16次、exp35_gen/loc的20/10-shot各8次，共32次原主线训练缺口；exp35_gen_de_vertigo 50-shot本地缺metric摘要。exp36_1四图四shot已登记完成，其原配方结果不改名为exp37。暂停的exp33/34/36不由本计划自动恢复。

## 2026-11-04：Unseen-4 当前结果分析

已完成三组未适配父模型的zero-shot，所有定位／离散结果覆盖8000条，连续覆盖5120帧。定位单任务XY=339.9282、joint=344.4904；生成单任务离散PSNR=11.3790、FID=77.4963，joint为11.0413／103.5831，表明这些Seen模型直接跨图仍有明显困难。joint的连续T-Warp／T-Diff较低（42.7217／48.6215 vs 44.5674／50.1917），但PSNR、LPIPS、FVD更差，不能只根据时序误差宣称生成整体更好。当前没有exp37的few-shot适配结果，下一步需补齐逐图400步后的query及Seen-retention，才能评估适配收益与遗忘；历史exp35／36结果不可充作exp37结果。
