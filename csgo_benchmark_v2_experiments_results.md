csgo benchmark v2 实验设计与表格维护规则
===========================================

## 1. exp31~exp36 实验设计定位

| 实验系列 | 设计定位 |
|---|---|
| `exp31` | Seen-10 full-head 实验；`exp31` 为 joint 联合训练且开启 aux_loc_loss 与 perception_loss，`exp31_1` 为关闭 aux 与 perception 的 loss-only control，`exp31_gen`/`exp31_loc` 为单任务实验，评测  Seen-10 与CrossMap zero-shot。 |
| `exp31*_3maps` | Benchmark v2 Seen-3 full-head 联合训练低成本 `2 x 2` loss ablation；固定使用 `de_ancient`、`de_dust2`、`de_nuke`，分别覆盖 aux+perception、joint-only、aux-only、perception-only，只评测 Seen-3。 |
| `exp32` | Seen-10 LoRA 实验；`exp32` 为 joint 联合训练且关闭 aux_loc_loss 与 perception_loss，`exp32_gen`/`exp32_loc` 为单任务实验，评测  Seen-10 与CrossMap zero-shot。 |
| `exp33`/`exp34` | 分别从 matching 的 `exp31*`/`exp32*` 初始化，在 CrossMap-4 四图共同 support 上进行非 map-specific 的 `100/50/20/10-shot` 适配；包含 joint、gen、loc 路由，并评测 CrossMap 与 Seen-10 retention。 |
| `exp35`/`exp36` | 分别从 matching 的 `exp31*`/`exp32*` 初始化，对 `cs_office`、`de_golden`、`de_palacio`、`de_vertigo` 四张 CrossMap 地图分别进行 map-specific 适配；shot 为 `100/50/20/10`，包含 joint、gen、loc 路由，评测 CrossMap 与 Seen-10 retention。 |
| `exp36_1` | 从 `exp31_1` 初始化的 map-specific full-head joint-only 对照；不存在 gen/loc 子训练路由，评测 CrossMap 与 Seen-10 retention。 |

结果表路由规则如下：joint 结果进入定位、离散生成、连续生成三个表；gen-only 结果进入两张生成表；loc-only 结果只进入定位表。map-specific 实验记录每个地图 checkpoint 的 CrossMap 结果和 Seen-10 retention；无 map 后缀的 family 名称只表示四个目标地图 CrossMap 结果的等地图宏平均，不是额外训练实验，也不要求对四个 retention checkpoint 再做 family 平均。非 map-specific 的 `exp33`/`exp34` 使用一个 checkpoint 同时覆盖 CrossMap-4，结果按 shot 记录跨地图聚合。

## 2. 实验归属与表格对应

- **ablation 3 maps实验进度表及ablation 3 maps表**：`exp31_3maps`、`exp31_1_3maps`、`exp31_2_3maps`、`exp31_3_3maps`。
- **主线进度表及主表**：`exp31*`、`exp32*`、`exp35*`、`exp36_1*`。
- **暂停支线进度表及补充表格**：`exp33*`、`exp34*`、`exp36*`。

上述归属是语义映射，不按实验名称字符串精确匹配。`exp31*_3maps` 虽然名称属于 `exp31*`，但必须进入独立的 ablation 表，不进入 Seen-10 主表。派生的宏平均、retention 结果以及按任务拆分的结果表行，都属于对应的同一个进度实验。

## 3. 排序保护

- 当前顺序是用户手工确认的 canonical order（规范顺序）。
- 更新实验结果时只能原位填写对应单元格，严格禁止全局排序、按名称排序、重建表、移动已有行、把空结果移到末尾或改变分隔行。
- ablation、主表和补充表格中的定位、离散生成、连续生成三个任务表，其现有相对顺序均不可改变。
- `ablation 3 maps表` 必须保留当前两行 Setext 一级标题格式（标题下一行 `====================`），以兼容已启动的旧版 map-specific runner；禁止改写为 `#` 标题。

## 4. 新实验加入流程

新增实验时，先确定 parent、训练路线（joint/gen/loc）、Seen 或 CrossMap、完整协议或固定地图子集、zero-shot 或 few-shot、是否 map-specific、地图、shot、评测任务、retention 和 macro；再判断其属于 ablation、主线还是暂停支线。随后按照当前最近的同族语义顺序，将实验加入对应的进度表和适用的任务结果表。map-specific 实验同时预留逐图结果和 CrossMap family macro；只添加必要的任务表，不制造不存在的 gen/loc 子实验。插入后核对三个任务表的语义投影、地图顺序、shot 顺序、retention 和宏平均。



# csgo benchmark v2 实验进度


## ablation 3 maps实验进度表

| 实验版本 | 训练 | 推理 | Metric 计算 |
|---|---|---|---|
| `exp31_3maps` | 未开始；目标 50 epochs，预计 step 5900 | 未开始；Seen-3 定位 6,000 + 离散生成 6,000 + 连续生成 3,840 | 未开始 |
| `exp31_1_3maps` | ✅ Seen-3，50 epochs，step 5850 | ✅ Seen-3：定位 6,000 + 离散生成 6,000 + 连续生成 3,840 | ✅ Seen-3：定位/离散/连续 metric 与三地图聚合完成 |
| `exp31_2_3maps` | 未开始；目标 50 epochs，预计 step 5900 | 未开始；Seen-3 定位 6,000 + 离散生成 6,000 + 连续生成 3,840 | 未开始 |
| `exp31_3_3maps` | ✅ Seen-3，50 epochs，step 5900 | ✅ Seen-3：定位 6,000 + 离散生成 6,000 + 连续生成 3,840 | ✅ Seen-3：定位/离散/连续 metric 与三地图聚合完成 |


## 主线实验进度表

| 实验版本 | 训练 | 推理 | Metric 计算 |
|---|---|---|---|
| `exp31` | Seen-10 未完成，step 10000/19550；✅ 当前评测 checkpoint-6000 | ✅ Seen-10：定位 20,000 + 离散生成 20,000 + 连续生成 12,800<br>CrossMap-4 zero-shot：未开始 | ✅ Seen-10：定位/离散/连续 metric 与聚合完成<br>CrossMap-4 zero-shot：未开始 |
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



ablation 3 maps表
====================


## 定位

| Setting | Task | Experiment | Shot/map | ablation | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Seen-3 ablation | Localization | exp31_3maps | dust2, ancient, nuke | aux-loc loss + perception loss |  |  |  |  | - |
| Seen-3 ablation | Localization | exp31_1_3maps | dust2, ancient, nuke | - - | 83.911 | 3.887 | 3.251 | 35.434 | 5850 |
| Seen-3 ablation | Localization | exp31_2_3maps | dust2, ancient, nuke | aux-loc loss |  |  |  |  | - |
| Seen-3 ablation | Localization | exp31_3_3maps | dust2, ancient, nuke | perception loss | 82.114 | 3.900 | 3.314 | 35.501 | 5900 |


## 离散生成

| Setting | Task | Experiment | Shot/map | ablation | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-3 ablation | Discrete generation | exp31_3maps | dust2, ancient, nuke | aux-loc loss + perception loss |  |  |  |  |  | - |
| Seen-3 ablation | Discrete generation | exp31_1_3maps | dust2, ancient, nuke | - - | 14.304 | 0.407 | 0.624 | 0.516 | 29.829 | 5850 |
| Seen-3 ablation | Discrete generation | exp31_2_3maps | dust2, ancient, nuke | aux-loc loss |  |  |  |  |  | - |
| Seen-3 ablation | Discrete generation | exp31_3_3maps | dust2, ancient, nuke | perception loss | 17.086 | 0.410 | 0.567 | 0.580 | 31.409 | 5900 |


## 连续生成

| Setting | Task | Experiment | Shot/map | ablation | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | FVD↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-3 ablation | Continuous generation | exp31_3maps | dust2, ancient, nuke | aux-loc loss + perception loss |  |  |  |  |  |  | - |
| Seen-3 ablation | Continuous generation | exp31_1_3maps | dust2, ancient, nuke | - - | 14.761 | 0.423 | 0.616 | 34.344 | 39.686 | 886.168 | 5850 |
| Seen-3 ablation | Continuous generation | exp31_2_3maps | dust2, ancient, nuke | aux-loc loss |  |  |  |  |  |  | - |
| Seen-3 ablation | Continuous generation | exp31_3_3maps | dust2, ancient, nuke | perception loss | 15.303 | 0.433 | 0.586 | 32.357 | 37.824 | 836.473 | 5900 |


# csgo benchmark v2 主表


## 定位

| Setting | Task | Experiment | Shot/map | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Seen-10 | Localization | exp31 | - | 195.048 | 10.489 | 4.344 | 79.060 | 6000 |
| CrossMap-4 zero-shot | Localization | exp31 | - |  |  |  |  | - |
| Seen-10 | Localization | exp31_1 | - | 82.750 | 4.354 | 2.969 | 37.130 | 19500 |
| CrossMap-4 zero-shot | Localization | exp31_1 | - | 330.421 | 17.483 | 3.895 | 89.844 | 19500 |
| Seen-10 | Localization | exp31_loc | - | 79.294 | 4.134 | 2.708 | 36.173 | 19500 |
| CrossMap-4 zero-shot | Localization | exp31_loc | - |  |  |  |  | - |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Seen-10 | Localization | exp32 | - | 211.975 | 10.766 | 4.652 | 81.964 | 6000 |
| Seen-10 | Localization | exp32 | - | 123.194 | 6.38 | 3.352 | 55.449 | 19500 |
| CrossMap-4 zero-shot | Localization | exp32 | - | 340.912 | 18.794 | 4.188 | 87.616 | 19500 |
| Seen-10 | Localization | exp32_loc | - | 48.961 | 3.006 | 2.970 | 26.047 | 19500 |
| CrossMap-4 zero-shot | Localization | exp32_loc | - |  |  |  |  | - |
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

| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Discrete generation | exp31 | - | 13.212 | 0.4054 | 0.6425 | 0.4972 | 36.066 | 6000 |
| CrossMap-4 zero-shot | Discrete generation | exp31 | - |  |  |  |  |  | - |
| Seen-10 | Discrete generation | exp31_1 | - | 14.954 | 0.4319 | 0.5725 | 0.5375 | 27.949 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp31_1 | - | 11.747 | 0.4156 | 0.7400 | 0.4522 | 96.250 | 19500 |
| Seen-10 | Discrete generation | exp31_gen | - | 14.561 | 0.4268 | 0.5888 | 0.5258 | 28.629 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp31_gen | - | 12.288 | 0.4380 | 0.7358 | 0.4542 | 101.199 | 19500 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Seen-10 | Discrete generation | exp32 | - | 12.995 | 0.3974 | 0.6583 | 0.4889 | 44.101 | 6000 |
| Seen-10 | Discrete generation | exp32 | - | 14.1528 | 0.4141 | 0.6035 | 0.5229 | 36.6005 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp32 | - | 11.5381 | 0.4075 | .7396 | 0.4471 | 88.0380 | - |
| Seen-10 | Discrete generation | exp32_gen | - | 13.965 | 0.4113 | 0.6106 | 0.5163 | 35.584 | 19500 |
| CrossMap-4 zero-shot | Discrete generation | exp32_gen | - | 11.081 | 0.3690 | 0.7210 | 0.4592 | 79.887 | 19500 |
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

| Setting | Task | Experiment | Shot/map | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | FVD↓ | Checkpoint step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Continuous generation | exp31 | - | 13.255 | 0.4026 | 0.6417 | 40.134 | 46.190 | 1011.744 | 6000 |
| CrossMap-4 zero-shot | Continuous generation | exp31 | - |  |  |  |  |  |  | - |
| Seen-10 | Continuous generation | exp31_1 | - | 15.571 | 0.4411 | 0.5446 | 28.879 | 35.688 | 674.180 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp31_1 | - | 11.826 | 0.4107 | 0.7421 | 37.947 | 44.492 | 1158.692 | 19500 |
| Seen-10 | Continuous generation | exp31_gen | - | 15.164 | 0.4349 | 0.5622 | 31.409 | 38.033 | 737.197 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp31_gen | - | 12.252 | 0.4291 | 0.7379 | 35.262 | 42.141 | 1131.397 | 19500 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen-10 | Continuous generation | exp32 | - | 13.078 | 0.3953 | 0.6601 | 40.511 | 46.401 | 1058.158 | 6000 |
| Seen-10 | Continuous generation | exp32 | - | 14.5725 | 0.4190 | 0.5826 | 34.0925 | 40.3596 | 848.4586 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp32 | - | 11.5966 | 0.4018 | 0.7452 | 44.2626 | 50.0603 | 1255.7838 | 19500 |
| Seen-10 | Continuous generation | exp32_gen | - | 14.342 | 0.4162 | 0.5929 | 35.698 | 41.867 | 848.071 | 19500 |
| CrossMap-4 zero-shot | Continuous generation | exp32_gen | - | 11.042 | 0.3628 | 0.7208 | 46.932 | 52.377 | 1349.984 | 19500 |
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
