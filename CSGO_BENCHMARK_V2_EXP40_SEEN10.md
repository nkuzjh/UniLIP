# CSGO Benchmark v2：exp39 αloc=2 与 exp40 Seen-10 Stage 2 消融

## 2026-11-13：设计与实验定位

本轮新增四个独立实验：从原始模型开始的 `exp39_consloc2`，以及从同一 `exp39@12000` 模型开始的 `exp40_s2_P4`、`exp40_s2_U2`、`exp40_s2_both`。用户已确认 `consloc2` 指**主定位损失 `alpha_loc_loss=2`**，不是辅助定位损失；该 Stage 1 实验的全部辅助目标仍关闭。三个 exp40 的主定位权重保持 **4**，不使用新 α2 实验作为来源。

四组正式终点均为累计 **19550次 optimizer 更新**，与 exp37／exp39 对齐；按既定比较口径，exp32_gen／loc 的19500与19550之间50步差异视为允许误差，表格保留各自实际步数。相同更新数不表示相同FLOPs或GPU时间。复用现有权重可减少本轮新增计算，但完整两阶段方法的成本仍包括前12000步。

用户决定将新增训练预算优先投入 exp39 基座，本轮不安排 exp38 四分支继续至final。已有 exp38 的4000步结果和恢复能力保留，不删除checkpoint，也不将缺少final描述为已经证明最终失败。此前[exp38／exp39结果与对照](CSGO_BENCHMARK_V2_EXP38_SEEN10.md)仍用于报告已观察到的取舍。

## 配置矩阵

| 实验 | 权重初始化 | Stage 1主αloc | Stage 2主αloc | perception终值／attention | aux-loc终值／candidate_tau | 本实验训练长度 | 累计final |
|---|---|---:|---:|---|---|---:|---:|
| `exp39_consloc2` | 原始UniLIP-1B＋π0.5；不加载exp39 | 2 | — | 关闭 | 关闭 | Stage 1 19550 | 19550 |
| `exp40_s2_P4` | `exp39/checkpoint-12000/model.safetensors` | 4 | 4 | 0.05／false | 关闭 | Stage 2 7550 | 12000＋7550＝19550 |
| `exp40_s2_U2` | 同一exp39@12000 | 4 | 4 | 关闭 | 1／1 | Stage 2 7550 | 12000＋7550＝19550 |
| `exp40_s2_both` | 同一exp39@12000 | 4 | 4 | 0.05／false | 1／1 | Stage 2 7550 | 12000＋7550＝19550 |

`P4`／`U2` 是3maps中辅助目标的配方来源，不是模型权重来源。P4使用当前定位头对应的teacher、vision-tower特征SmoothL1、teacher detach、无EMA、`loc_perception_use_attention_weight=false`、`linear_1m_sigma`时间权重。U2使用combined EM＋uncertainty、2候选、candidate_tau=1、unc_tau=1、unc_min_weight=0.05、共享定位噪声、`exp_sigma` λ=5、不额外renormalize。

## 训练、保存与评测节点

| 设置 | exp39_consloc2 | 三组exp40 |
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
| 完整checkpoint | 12000、16000、19550；keep3 | 4000、7550；keep2 |
| 根目录重复模型导出 | 关闭 | 关闭 |
| 默认训练行为 | 一次训练到final；可在12000／16000暂停 | 首次默认4000停止；`--stop-after-step 0`可一次完成7550 |
| validation／test评测节点 | 16000、final=19550；12000仅保存供未来分叉 | local4000、final=7550，分别对应累计16000／19550 |

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

1. `scripts/run_exp37_seen10.py`注册`exp39_consloc2`，保持原exp37／exp39／exp37_s2行为；新配置相对exp39只改变实验身份与主αloc。
2. 新增独立`scripts/run_exp40_seen10.py`和三份同名YAML，提供validate／train／infer／metrics及自身恢复；保留旧exp38入口和历史实验身份。
3. 新入口仅在显式`--execute`时写运行快照或启动子进程；只读validate／dry run不启动训练、推理、评测。
4. CPU验证覆盖来源身份、主／辅助目标路由、7550调度、252／503缩放、精确保存、完整恢复、评测节点、batch／split路由及minimal／full切换。真实来源检查不加载大模型到GPU。
5. 可执行命令集中在[record.md新增章节](record.md#exp39-exp40-seen10-20261113)，三个exp40均将4000阶段和恢复final分开，是否继续由用户决定。

## 实验进度与结果记录

代码、配置和手动命令已实现并完成CPU验收。当前没有本轮四组新实验的实测结果；不把配置或CPU检查视为训练／评测完成。

| 实验 | 训练状态 | 保存节点 | validation节点 | test节点 | 结果 |
|---|---|---|---|---|---|
| `exp39_consloc2` | 待用户手动启动 | 12000／16000／19550 | 16000／19550 | 16000／19550 | — |
| `exp40_s2_P4` | 待用户手动启动 | local4000／7550 | local4000／7550 | local4000／7550 | — |
| `exp40_s2_U2` | 待用户手动启动 | local4000／7550 | local4000／7550 | local4000／7550 | — |
| `exp40_s2_both` | 待用户手动启动 | local4000／7550 | local4000／7550 | local4000／7550 | — |

### 实现验收（2026-11-13）

- 新增[exp39_consloc2配置](csgo_configs/exp39_consloc2.yaml)及exp37入口注册；三组Stage 2配置分别为[P4](csgo_configs/exp40_s2_P4.yaml)、[U2](csgo_configs/exp40_s2_U2.yaml)、[both](csgo_configs/exp40_s2_both.yaml)，由[独立exp40入口](scripts/run_exp40_seen10.py)管理。旧exp38入口未修改。
- exp37／exp39入口、旧exp38入口、新exp40入口和精确保存／暂停恢复组件共**75项CPU测试通过**。覆盖错误来源拒绝、主／辅助配方、固定horizon、自身完整恢复、评测节点、validation／test路由、推理batch与产物身份。保存组件使用临时目录中的合成小模型验证，不是本项目实验训练。
- record新增9个bash片段及其中7个nohup主体均通过`bash -n`；42次CLI参数解析通过。原record内容及CRLF换行保持不变。新实验的4000与final阶段为独立手动片段。
- 真实exp39@12000仅做CPU只读预检：模型SHA-256为`d1cb5c67aa0cc2103fca517c8ebdd622bc70db0ce96a8582fd0d7c55d6c3317f`，2830个张量、1128个LoRA张量，Trainer为12000/19550；未将大模型载入GPU。首次实际执行仍由入口重新检查并绑定来源。
- 未启动任何新实验的训练、推理或指标计算；训练／结果目录不由文档和dry run创建。GPU可运行性和实际指标需由用户在远端执行后确认。

## 对资源转向的判断

将后续预算转向exp39基座是合理的搜索决策。当前同累计16000步，exp39的离散五项指标均优于exp38各分支，且多数定位／连续指标更好；α4是值得继续检验的Stage 1设置。此前优先完成exp38 final的建议，主要服务于完成原路线的固定终点对照；当前可以优先追求新基座上的完整方案效果。

但“exp38完全落后”不符合全部指标：exp38 joint α8／4／12的Z分别为2.974／3.030／3.016，优于exp39的3.072；Stage 2 α12的Pitch为2.667、FVD为937.934，也优于exp39的2.883／959.891。应表述为“exp39整体更有吸引力，因此不再优先追加exp38计算”，而非“已证明exp38 final无价值”。这些是已观察到的4000／16000节点取舍，不是对未运行final的结论。

本轮三组exp40可比较辅助配方，并判断其完整两阶段路线相对exp39单阶段、exp37与单任务模型的最终表现；`both`对`P4`／`U2`还能衡量加入另一项辅助的条件增量。但exp40与单阶段exp39之间同时改变了学习率路径和optimizer重置，不能将全部差异写成辅助loss净收益。若最终需要严格证明“辅助＋联合优于同基座无辅助联合续训”，后续应补同来源、同α4、同7550调度的无辅助续训对照。本轮按用户要求只新增上述四组，不额外创建第五个训练实验。
