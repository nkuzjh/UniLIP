# CSGO Benchmark v2：exp39 αloc=2 与 exp40 Seen-10 Stage 2 消融

## 2026-11-13：设计与实验定位

首批四个独立实验是从原始模型开始的 `exp39_consloc2`，以及从同一 `exp39@12000` 模型开始的 `exp40_s2_P4`、`exp40_s2_U2`、`exp40_s2_both`。用户已确认 `consloc2` 指**主定位损失 `alpha_loc_loss=2`**，不是辅助定位损失；该 Stage 1 实验的全部辅助目标仍关闭。原三个 exp40 的主定位权重保持 **4**，不使用新 α2 实验作为来源。随后批准新增从`exp39@16000`开始的`exp40_1_s2_both`，其6000步、更高学习率的独立设置见[新增方案](#exp40-1-s2-both-20261113)。

首批四组正式终点均为累计 **19550次 optimizer 更新**，与 exp37／exp39 对齐；按既定比较口径，exp32_gen／loc 的19500与19550之间50步差异视为允许误差，表格保留各自实际步数。新增`exp40_1_s2_both`累计为**22000步**，单独标注为额外预算探索组。相同更新数不表示相同FLOPs或GPU时间。复用现有权重可减少本轮新增计算，但完整两阶段方法成本仍包括各自Stage 1前缀。

用户决定将新增训练预算优先投入 exp39 基座，本轮不安排 exp38 四分支继续至final。已有 exp38 的4000步结果和恢复能力保留，不删除checkpoint，也不将缺少final描述为已经证明最终失败。此前[exp38／exp39结果与对照](CSGO_BENCHMARK_V2_EXP38_SEEN10.md)仍用于报告已观察到的取舍。

## 配置矩阵

| 实验 | 权重初始化 | Stage 1主αloc | Stage 2主αloc | perception终值／attention | aux-loc终值／candidate_tau | 本实验训练长度 | 累计final |
|---|---|---:|---:|---|---|---:|---:|
| `exp39_consloc2` | 原始UniLIP-1B＋π0.5；不加载exp39 | 2 | — | 关闭 | 关闭 | Stage 1 19550 | 19550 |
| `exp40_s2_P4` | `exp39/checkpoint-12000/model.safetensors` | 4 | 4 | 0.05／false | 关闭 | Stage 2 7550 | 12000＋7550＝19550 |
| `exp40_s2_U2` | 同一exp39@12000 | 4 | 4 | 关闭 | 1／1 | Stage 2 7550 | 12000＋7550＝19550 |
| `exp40_s2_both` | 同一exp39@12000 | 4 | 4 | 0.05／false | 1／1 | Stage 2 7550 | 12000＋7550＝19550 |
| `exp40_1_s2_both` | `exp39/checkpoint-16000/model.safetensors` | 4 | 4 | 0.05／false | 1／1 | Stage 2 6000 | **16000＋6000＝22000** |

`P4`／`U2` 是3maps中辅助目标的配方来源，不是模型权重来源。P4使用当前定位头对应的teacher、vision-tower特征SmoothL1、teacher detach、无EMA、`loc_perception_use_attention_weight=false`、`linear_1m_sigma`时间权重。U2使用combined EM＋uncertainty、2候选、candidate_tau=1、unc_tau=1、unc_min_weight=0.05、共享定位噪声、`exp_sigma` λ=5、不额外renormalize。

## 训练、保存与评测节点

| 设置 | exp39_consloc2 | 原三组exp40（exp39@12000来源） |
|---|---|---|
| 初始化方式 | 与exp39相同的原始初始化，独立训练 | 只加载exp39@12000模型权重，新建optimizer／scheduler／RNG，local step从0开始 |
| optimizer | AdamW；β=(0.9,0.999)、eps=1e-8、WD=0、clip norm=1 | 相同 |
| 普通参数／活动LoRA峰值LR | 1e-4 | 2e-5 |
| action connector／norm／projector峰值LR | 5e-4 | 1e-4 |
| scheduler | cosine_with_min_lr；继承exp39的min_lr=1e-5 | cosine_with_min_lr；min_lr_rate=0.1 |
| 学习率warmup | ratio=0.003，19550步对应59步 | **252步** |
| 辅助权重ramp | 无 | 启用的目标在local0～503线性升至矩阵终值，之后恒定 |
| 源样本有效batch／seed | 128／42 | 128／42 |
| 总调度长度 | 固定19550，epoch参数保持50 | 固定7550，epoch参数保持50；max_steps优先 |
| 完整checkpoint | 每4000步保存：4000、8000、12000、16000，另存final=19550；最多保留最近2份（2026-11-16更新） | 4000、7550；keep2 |
| 根目录重复模型导出 | 关闭 | 关闭 |
| 默认训练行为 | 一次训练到final；可在4000／8000／12000／16000暂停 | 首次默认4000停止；`--stop-after-step 0`可一次完成7550 |
| validation／test评测节点 | 16000、final=19550；4000／8000／12000仅作中间保存，随新checkpoint滚动清理 | local4000、final=7550，分别对应累计16000／19550 |

`exp39_consloc2`正常保存final后仅保留16000和19550两个完整checkpoint；12000不再长期保留。正在运行的进程不会热更新保存策略，需在使用新入口启动或恢复时生效；已错过的保存节点不会补回。

旧的三节点／keep3运行记录仍可评测；训练入口在严格核对配置与运行参数后，仅迁移保存策略，完整恢复optimizer／scheduler／RNG及19550步调度。dry run不改写运行记录。2026-11-16保存策略变更已通过54项CPU测试，其中小模型实测五次保存与暂停恢复后只保留最后两个完整checkpoint；本次未执行项目实验或清理既有权重。

Stage 2缩放沿用既有两阶段关系：以6000步版本的warmup200、ramp400为基准，分别取 `round(200×7550/6000)=252` 与 `round(400×7550/6000)=503`。不能将ramp误写为2×252。在4000暂停时，scheduler仍按7550步计算；之后完整恢复本实验optimizer／scheduler／RNG，继续剩余3550步，不重新warmup，不重复加载Stage 1权重。

已有来源已做只读身份核查：exp39@12000的Trainer为12000/19550，模型大小4,577,302,678字节；保存配方为主α4、全部辅助关闭。首次实际训练前，入口继续检查来源的实验身份、配方、checkpoint step、结构及SHA-256，并要求三个分支绑定同一来源。首次初始化不要求exp39完成final；本实验恢复必须使用自己的完整checkpoint，不能跨P4／U2／both恢复。

## 数据与推理口径

使用现有完整Seen-10：50000条训练、5000条validation、20000条离散test、12800帧连续test。地图、数据划分和benchmark指标保持不变。

- validation的`--task all`只包含定位和离散生成；test的`--task all`包含定位、离散生成和连续生成。连续生成仅接受test。
- 新实验固定默认定位推理batch32、生成batch16，seed42；metrics使用相同batch参数定位对应产物，生成指标计算batch仍为1。支持单独选择`localization`、`gen`、`gen_continuous`。
- 每个实验／checkpoint／split／seed／batch独立保存推理配置、manifest与结果，按十图等权宏平均汇总。生成沿用现有推理模型模式；与已有结果比较时保持相同batch及推理设置，不将batch差异归因于辅助目标。
- 16000／4000是中间报告节点，final为固定完整终点。新的超参选择使用现有validation；test完整报告指定节点，不拼接各checkpoint最优指标。既有test结果参与了本轮探索方向的提出，论文中不将它描述为完全未观察的调参过程。
- 配置默认minimal资产。首次train将`--asset-mode minimal`替换为`--asset-mode full --data-dir /path/to/preprocessed_data`即可切换全量资产；这只改变资产读取路径，不改变样本集合。resume／infer／metrics继承已保存的资产配置，不中途切换后端。

## 实现范围与验收要求

1. `scripts/run_exp37_seen10.py`注册`exp39_consloc2`，保持原exp37／exp39／exp37_s2行为；新配置相对exp39改变实验身份与主αloc，2026-11-16再将本组保存节点扩为每4000步加final、最多保留最近2份，训练目标和学习率调度不变。
2. 新增独立`scripts/run_exp40_seen10.py`和三份同名YAML，提供validate／train／infer／metrics及自身恢复；保留旧exp38入口和历史实验身份。
3. 新入口仅在显式`--execute`时写运行快照或启动子进程；只读validate／dry run不启动训练、推理、评测。
4. CPU验证覆盖来源身份、主／辅助目标路由、7550调度、252／503缩放、精确保存、完整恢复、评测节点、batch／split路由及minimal／full切换。真实来源检查不加载大模型到GPU。
5. 可执行命令集中在[record.md新增章节](record.md#exp39-exp40-seen10-20261113)，三个exp40均将4000阶段和恢复final分开，是否继续由用户决定。

## 实验进度与结果记录

下表记录截至2026-11-19的本机同步状态。定位与生成的推理 batch 分别为32和16，seed均为42。当前实测指标和来源见[同步结果](#exp39-consloc2-results-20261119)。

| 实验 | 训练状态 | 保存节点 | validation节点 | test节点 | 结果 |
|---|---|---|---|---|---|
| `exp39_consloc2` | 19550/19550，epoch50 | 本机有16000／19550权重和状态，keep2 | 两节点定位／离散均已同步 | 两节点三项均已同步 | [十份实测摘要](#exp39-consloc2-results-20261119) |
| `exp40_s2_P4` | 本机未见训练产物 | 计划local4000／7550 | 两节点摘要未见 | 两节点摘要未见 | — |
| `exp40_s2_U2` | 本机未见训练产物 | 计划local4000／7550 | 两节点摘要未见 | 两节点摘要未见 | — |
| `exp40_s2_both` | 本机未见训练产物 | 计划local4000／7550 | 两节点摘要未见 | 两节点摘要未见 | — |
| `exp40_1_s2_both` | 本机local2000/6000，累计18000 | 本机有local2000，计划local4000／6000 | 11-18远端截图显示2000两项完成，本机摘要未见 | 2000三项已同步。4000／6000摘要未见 | [2000 test](#exp40-1-s2-both-test-2000-20261119) |

### 实现验收（2026-11-13）

- 新增[exp39_consloc2配置](csgo_configs/exp39_consloc2.yaml)及exp37入口注册；三组Stage 2配置分别为[P4](csgo_configs/exp40_s2_P4.yaml)、[U2](csgo_configs/exp40_s2_U2.yaml)、[both](csgo_configs/exp40_s2_both.yaml)，由[独立exp40入口](scripts/run_exp40_seen10.py)管理。旧exp38入口未修改。
- exp37／exp39入口、旧exp38入口、新exp40入口和精确保存／暂停恢复组件共**75项CPU测试通过**。覆盖错误来源拒绝、主／辅助配方、固定horizon、自身完整恢复、评测节点、validation／test路由、推理batch与产物身份。保存组件使用临时目录中的合成小模型验证，不是本项目实验训练。
- record新增9个bash片段及其中7个nohup主体均通过`bash -n`；42次CLI参数解析通过。原record内容及CRLF换行保持不变。新实验的4000与final阶段为独立手动片段。
- 真实exp39@12000仅做CPU只读预检：模型SHA-256为`d1cb5c67aa0cc2103fca517c8ebdd622bc70db0ce96a8582fd0d7c55d6c3317f`，2830个张量、1128个LoRA张量，Trainer为12000/19550；未将大模型载入GPU。首次实际执行仍由入口重新检查并绑定来源。
- 未启动任何新实验的训练、推理或指标计算；训练／结果目录不由文档和dry run创建。GPU可运行性和实际指标需由用户在远端执行后确认。

## 对资源转向的判断（2026-11-13设计阶段）

将后续预算转向exp39基座是合理的搜索决策。当前同累计16000步，exp39的离散五项指标均优于exp38各分支，且多数定位／连续指标更好；α4是值得继续检验的Stage 1设置。此前优先完成exp38 final的建议，主要服务于完成原路线的固定终点对照；当前可以优先追求新基座上的完整方案效果。

但“exp38完全落后”不符合全部指标：exp38 joint α8／4／12的Z分别为2.974／3.030／3.016，优于exp39的3.072；Stage 2 α12的Pitch为2.667、FVD为937.934，也优于exp39的2.883／959.891。应表述为“exp39整体更有吸引力，因此不再优先追加exp38计算”，而非“已证明exp38 final无价值”。这些是已观察到的4000／16000节点取舍，不是对未运行final的结论。

原三组exp40可比较辅助配方，并判断其完整两阶段路线相对exp39单阶段、exp37与单任务模型的最终表现；`both`对`P4`／`U2`还能衡量加入另一项辅助的条件增量。但exp40与单阶段exp39之间同时改变了学习率路径和optimizer重置，不能将全部差异写成辅助loss净收益。若最终需要严格证明“辅助＋联合优于同基座无辅助联合续训”，后续应补同来源、同α4、同Stage 2调度的无辅助续训对照。新增`exp40_1_s2_both`同样开启辅助；本次仍未创建无辅助Stage 2对照。

<a id="exp40-1-s2-both-20261113"></a>

## 2026-11-13：exp40_1_s2_both，从exp39@16000继续6000步

本方案已经用户审核通过。新增[配置](csgo_configs/exp40_1_s2_both.yaml)，仍由[exp40入口](scripts/run_exp40_seen10.py)管理，使用独立实验目录。已有三组的12000来源、7550步调度、学习率、保存节点及恢复身份不变。完整串行和中断恢复命令见[record.md](record.md#exp40-1-s2-both-20261113)。

### 来源、目标与超参

只加载`outputs/csgo_1b/exp39/checkpoint-16000/model.safetensors`。它来自原19550步Stage 1调度的第16000步，不另训一个总长度16000的Stage 1。首次Stage 2新建AdamW、scheduler、RNG及local step；后续仅恢复本组完整状态。主目标为`L_gen＋4×L_loc`，辅助为P4＋U2：perception终值0.05、attention=false，aux-loc终值1、combined、2候选、candidate_tau=1；其他辅助细节、LoRA及冻结范围继承原both。

| 项目 | exp40_s2_both（原方案） | exp40_1_s2_both（新增） |
|---|---|---|
| Stage 1来源 | exp39@12000 | **exp39@16000** |
| Stage 2／累计final | 7550／19550 | **6000／22000** |
| 普通参数／活动LoRA峰值LR | 2e-5 | **3e-5** |
| action connector／norm／projector峰值LR | 1e-4 | **1.5e-4** |
| 学习率warmup | 252步 | **200步** |
| 辅助权重ramp | 0～503 | **0～400**，两项分别从0升至0.05／1 |
| scheduler | cosine_with_min_lr，min_lr_rate=0.1 | 相同；两类最终LR为3e-6／1.5e-5 |
| optimizer | AdamW，β=(0.9,0.999)，eps=1e-8，WD=0，clip norm=1 | 相同 |
| 有效基础batch／seed | 128／42 | 相同 |
| 完整checkpoint／保留数量 | 4000、7550／2 | **2000、4000、6000／3** |
| 默认首次暂停 | local4000 | **local2000**；`--stop-after-step 0`可完整训练6000 |
| 根目录重复模型导出 | 关闭 | 关闭 |

新学习率是原Stage 2的1.5倍，保持5倍组间比例；200／400分别恢复6000步版本的约3.33% warmup和6.67%辅助ramp。exp39@16000的日志学习率约1.72e-5，本组通过warmup重新升至3e-5；这是已批准的探索取值，尚未证明优于旧配方。

### 节点与执行行为

| local step | 累计step | 保存 | validation | test |
|---:|---:|---|---|---|
| 2000 | 18000 | 完整模型及optimizer／scheduler／RNG／Trainer | 定位＋离散生成 | 定位＋离散＋连续生成 |
| 4000 | 20000 | 同上 | 同上 | 同上 |
| 6000／final | 22000 | 同上 | 同上 | 同上 |

提供一段手动启动的串行命令：训练到2000并评测，再完整恢复到4000并评测，最后恢复到6000并评测。中间暂停不压缩6000步scheduler，不重新warmup、不再次加载Stage 1权重。失败时由`set -euo pipefail`停止该段队列，用户从最后保存的本实验checkpoint恢复；训练已到6000时只补缺失评测。

默认定位推理batch32、生成batch16、seed42，指标计算batch1；minimal／full切换沿用前文。所有节点使用独立checkpoint／split／seed／batch目录；validation的all不包含连续生成，test的all包含三项。`final`在本实验中为6000，原三个exp40仍为7550。

### 对照与工程约束

- 完整路径比19550多2450更新（12.53%），比19500多2500更新（12.82%）。该差异不属于既定50步容差；表格明确写累计22000，不能称作与19500／19550等更新预算。已有16000权重可复用，本轮实际新增训练为6000步；完整方法仍计入前缀成本。
- 相对原both同时改变来源节点、学习率、Stage 2长度和累计预算，结果评价整体训练路线；不能单独归因于辅助或更高LR。若后续需要严格辅助归因，需另设同16000来源、同6000调度的无辅助对照，本次未新增。
- 入口按实验管理来源、horizon、LR、warmup、ramp和checkpoint列表。来源hash只在使用同一Stage 1 checkpoint的分支间要求一致：原三组绑定exp39@12000，新组独立绑定exp39@16000；两组来源可以共存，不能跨来源或跨实验resume。
- 入口在手动传入`--execute`时执行训练、推理或评测。已同步的训练和评测状态见[2000步结果](#exp40-1-s2-both-test-2000-20261119)。

### 新增实验实现验收（2026-11-13）

- exp37／exp39入口、exp40入口和保存／暂停恢复组件共**64项CPU测试通过**，其中exp40为14项；验证固定6000调度、完整恢复、来源分组、评测节点和batch产物身份。组件测试仅使用临时目录中的合成小模型。
- 对修改前保存的原三组exp40共12种训练计划逐项比对，配置、命令、runtime和fingerprint全部一致，覆盖默认、完整训练、full资产和双卡设置。
- 真实exp39@16000只读预检通过：Trainer为16000/19550，2830个张量、1128个LoRA张量，SHA-256为`37060de430c54a201ca304bdd20992595de2d4b58eadd07c3c3060cbd1fdc5a3`。首次实际执行仍重新校验并绑定来源；本次未载入GPU。
- record新增4个bash片段及2个nohup主体均通过语法检查；以占位函数模拟首次与三个恢复节点，确认2000→4000→6000顺序、恢复标志、各split与batch参数。未执行真实实验命令，原record字节和CRLF保持不变。
- 扩展至旧exp38的83项测试时，旧exp38两个测试方法中的3个子用例报错：其`JOINT_EXPERIMENTS`已包含`exp38_s2_both`，使both误入“禁止辅助loss”检查。已核对旧入口及测试与当前HEAD一致，本次未修改；该问题不影响独立exp40入口，但不能将扩展测试记为全部通过。
- 新实验的训练、定位和生成结果目录均未创建；GPU端训练／推理／评测尚未执行。

<a id="exp39-consloc2-results-20261119"></a>

## 2026-11-19：exp39_consloc2 实测结果

训练状态为19550/19550、epoch50，本机保留16000和19550两个完整模型。它从原始 UniLIP＋π0.5 初始化，主定位权重为2，所有辅助目标关闭。与 `exp39` 的保存配方相比，训练目标只改变主定位权重，另外有实验名和保存节点／轮换设置差异。实际训练均为两卡×micro64×GAS1，有效基础 batch128、seed42。训练来源见[本组计划](outputs/csgo_1b/exp39_consloc2/experiment_plan.json)及[最终 Trainer 状态](outputs/csgo_1b/exp39_consloc2/checkpoint-19550/trainer_state.json)。

以下均为 Seen-10 十图等权宏平均。已核对十份摘要的 checkpoint、split、seed、batch、样本数及逐图指标，重算宏平均与摘要一致。定位推理 batch32、10步，生成 batch16、20步，均使用 minimal 资产。生成指标为 CUDA 上的 `benchmark_v2_core`，计算 batch1、paired_size448，连续 FVD 使用 clip length16、stride16、size224。

### 定位

| Split | Checkpoint | 样本数 | 推理 batch | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 摘要 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| validation | 16000 | 5000 | 32 | 105.445 | 5.548 | 3.530 | 44.022 | [JSON](outputs_loc/benchmark_v2/exp39_consloc2/checkpoint-16000/validation/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) |
| validation | 19550 | 5000 | 32 | 94.558 | 5.115 | 3.191 | 40.600 | [JSON](outputs_loc/benchmark_v2/exp39_consloc2/checkpoint-19550/validation/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) |
| test | 16000 | 20000 | 32 | 116.252 | 5.831 | 3.416 | 49.567 | [JSON](outputs_loc/benchmark_v2/exp39_consloc2/checkpoint-16000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) |
| test | 19550 | 20000 | 32 | 105.651 | 5.415 | 3.096 | 46.347 | [JSON](outputs_loc/benchmark_v2/exp39_consloc2/checkpoint-19550/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) |

### 离散生成

| Split | Checkpoint | 样本数 | 推理 batch | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 摘要 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| validation | 16000 | 5000 | 16 | 14.407 | 0.431 | 0.597 | 0.523 | 62.774 | [JSON](outputs_eval/benchmark_v2/exp39_consloc2/checkpoint-16000/validation/seed_42/infer_bs16/discrete/summary.json) |
| validation | 19550 | 5000 | 16 | 14.570 | 0.430 | 0.589 | 0.530 | 62.436 | [JSON](outputs_eval/benchmark_v2/exp39_consloc2/checkpoint-19550/validation/seed_42/infer_bs16/discrete/summary.json) |
| test | 16000 | 20000 | 16 | 14.066 | 0.417 | 0.611 | 0.516 | 36.283 | [JSON](outputs_eval/benchmark_v2/exp39_consloc2/checkpoint-16000/test/seed_42/infer_bs16/discrete/summary.json) |
| test | 19550 | 20000 | 16 | 14.189 | 0.416 | 0.604 | 0.521 | 35.227 | [JSON](outputs_eval/benchmark_v2/exp39_consloc2/checkpoint-19550/test/seed_42/infer_bs16/discrete/summary.json) |

### 连续生成 test

| Checkpoint | 帧数 | 推理 batch | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | TWE↓ | TDE↓ | OFE↓ | FVD↓ | 摘要 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16000 | 12800 | 16 | 14.420 | 0.422 | 0.593 | 0.530 | 34.471 | 40.709 | 17.222 | 888.619 | [JSON](outputs_eval/benchmark_v2/exp39_consloc2/checkpoint-16000/test/seed_42/infer_bs16/continuous/summary.json) |
| 19550 | 12800 | 16 | 14.569 | 0.420 | 0.586 | 0.535 | 34.002 | 40.264 | 17.252 | 865.734 | [JSON](outputs_eval/benchmark_v2/exp39_consloc2/checkpoint-19550/test/seed_42/infer_bs16/continuous/summary.json) |

两节点六项生成评测的60份逐图 JSON 均记录匹配率100%，缺失和额外预测均为0。预测图片同步期间曾只见部分文件，本节覆盖数取自完整的评测 JSON。两节点 test 数值见[16000步对照](CSGO_BENCHMARK_V2_EXP38_SEEN10.md#exp38-exp39-test-16000)和[final test对照](CSGO_BENCHMARK_V2_EXP38_SEEN10.md#exp38-exp39-final-20261112)。

<a id="exp40-1-s2-both-test-2000-20261119"></a>

## 2026-11-19：exp40_1_s2_both local2000 test

本机同步了 Stage 2 local2000、累计18000步的 checkpoint、状态及三项 test 摘要。计划绑定 `exp39@16000`，主 αloc=4，P4＋U2，实际两卡×micro4×GAS16，有效基础 batch128。test 定位／生成推理 batch32／16、seed42。三项均覆盖十图，已逐图复核。来源见[训练计划](outputs/csgo_1b/exp40_1_s2_both/experiment_plan.json)及[Trainer 状态](outputs/csgo_1b/exp40_1_s2_both/checkpoint-2000/trainer_state.json)。

| 任务 | 样本数 | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 摘要 |
|---|---:|---:|---:|---:|---:|---|
| 定位 | 20000 | 49.388 | 3.368 | 3.028 | 26.865 | [JSON](outputs_loc/benchmark_v2/exp40_1_s2_both/checkpoint-2000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) |

| 任务 | 样本数 | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 摘要 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 离散生成 | 20000 | 13.593 | 0.404 | 0.624 | 0.510 | 37.657 | [JSON](outputs_eval/benchmark_v2/exp40_1_s2_both/checkpoint-2000/test/seed_42/infer_bs16/discrete/summary.json) |

| 任务 | 帧数 | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | TWE↓ | TDE↓ | OFE↓ | FVD↓ | 摘要 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 连续生成 | 12800 | 13.867 | 0.407 | 0.611 | 0.521 | 37.455 | 43.531 | 17.496 | 963.824 | [JSON](outputs_eval/benchmark_v2/exp40_1_s2_both/checkpoint-2000/test/seed_42/infer_bs16/continuous/summary.json) |

<a id="exp38-exp39-exp40-analysis-20261119"></a>

## 2026-11-19：exp38／exp39／exp40 分析

**exp39 的 αloc=4 仍是当前最均衡的无辅助联合基座。** 同为累计16000步、定位／生成 batch32／16 时，α4 的四项定位宏平均均优于 α2、α8（`exp37`）和 α12。相对 α8，α4 还改善了离散五项和连续六项。α2 的 XY 为116.252，α4 为49.459，误差增加135.05%。Z 和 Yaw 分别增加89.79%和84.17%，三项在十图均变差。到19550步，α2 的 validation XY 从105.445降至94.558，test XY 从116.252降至105.651，四个定位误差在两个 split 上均继续下降。同期 α4 的 test XY 已为47.050，α2 仍高124.55%。α12 的 final XY 为95.145，同样远高于 α4。α8 的 final 为历史 batch1，四种权重共同比较以16000步为准。

**αloc=2 在两个节点都以定位性能换来了更好的生成表现。** 同16000步相对 α4，离散4/5项改善，FID 从35.956升至36.283。连续八项均改善，FVD 从959.891降至888.619（−7.43%，十图均改善）。相对同节点生成单任务 `exp32_gen`，α2 在离散4/5、连续7/8项占优，FID、FVD 偏高，其中 FVD 为888.619比867.691。到19550步，在 batch32／16的 α2／4／12 三组对照中，α2 拿到13项生成指标中的12项最优，OFE 由 α4 最优。α2 的离散 PSNR／FID 为14.189／35.227，连续 FVD 为865.734，低于 α4 的921.205（十图均下降，宏平均−6.02%）和 α12 的879.816。自身从16000到final的 test 生成有10/13项改善，validation 离散有4/5项改善。α2／α12 保留为权重消融，联合基座优先 α4。源数值见[累计16000步对照](CSGO_BENCHMARK_V2_EXP38_SEEN10.md#exp38-exp39-test-16000)及[final对照](CSGO_BENCHMARK_V2_EXP38_SEEN10.md#exp38-exp39-final-20261112)。

同16000步的单任务对照中，α4 联合模型的 XY、Pitch 优于 `exp32_loc`，Z、Yaw 略高。其离散 SSIM、FID 优于 `exp32_gen`，连续八项则均较差。当前实测给出了定位与生成之间的取舍，距离联合模型同时追平两个单任务仍有差距。详见[同累计步数三任务表](CSGO_BENCHMARK_V2_EXP38_SEEN10.md#exp38-exp39-test-16000)。

**exp38 的低学习率续训在累计16000步呈现局部收益。** 三个无辅助分支从同一 `exp37@12000` 续训。local4000 时，α4 相对 α8 改善 XY、Pitch、Yaw 和 FVD，Z 从2.974升至3.030。α12 相对 α8 改善 XY（52.309→50.624）、Pitch（3.236→2.667）和 FVD（973.564→937.934），Z 和 Yaw 走高。Stage 1 全程 α12 的大幅定位退化在这条晚期调权路线的观察点没有复现。保持 α8 的续训与单阶段 `exp37@16000` 相比，XY、Z、Yaw 改善，连续八项较差。单阶段 α4 的离散五项均优于同 batch 的三个 exp38 无辅助分支，XY 和 Yaw 也较低，这三个分支的 Z 更低。`exp38_s2_both` 对应的 FVD 为965.626，对照 joint 为973.564，同时 XY 从52.309升至54.076。两组推理 batch 分别为16／8和32／16，生成清单记录了活跃 Dropout，辅助损失的独立效应仍需同 batch 比较。四组 final 摘要尚缺，当前结论限于 local4000。详见[同累计步数三任务表](CSGO_BENCHMARK_V2_EXP38_SEEN10.md#exp38-exp39-test-16000)。

**exp40_1_s2_both 的已测2000步暂未表现出跨任务的净改善。** 对其实际起点 `exp39@16000`，XY（49.459→49.388）与 Yaw（26.914→26.865）变化很小，Z 从3.072升至3.368（+9.63%，十图均变差），Pitch 从2.883升至3.028。离散和连续的 LPIPS 及边界 F1 略有改善，离散 FID 从35.956升至37.657，连续 FVD 从959.891升至963.824。已经完成的 `exp39@19550` 在这17项指标均更好，但二者预算相差1550步。此对照反映从 α4 基座继续训练2000步后的整体变化，涉及重新设置学习率、重建优化器及新增辅助目标。同源、同调度的无辅助 Stage 2 对照仍是判断 P4＋U2 增量所需的证据。原三组 `exp40_s2_P4`、`exp40_s2_U2`、`exp40_s2_both` 暂无本机结果摘要，三种辅助配置当前无法排序。

后续先同步 `exp40_1_s2_both` 的 local2000 validation，已有续训按既定4000／6000节点观察，报告累计20000／22000步。若重点是论文的约19550步对照，优先做从 `exp39@12000` 起步的 `exp40_s2_both` 与同源同调度的无辅助联合续训，按local4000／7550成对评测。后续超参筛选依据 validation，test 按预定节点汇报。
