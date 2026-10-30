# J8 两阶段辅助微调设计

设计日期：2026-10-17（当时实验尚未启动）。结果更新：2026-10-22，S0_J8～S3_J8 均已完成 local1800，local1200／1800 两任务 validation 结果已登记，见 [结果与分析](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#stage2-results-20261022)。本文件以下配方与验收记录保留设计时口径；实际双卡 microbatch/GAS 不同，名义有效 batch128 相同但尾批造成实际曝光略不等，不能把这批结果视为严格只改变辅助 loss 的对照。执行命令集中放在 [record.md](record.md)，状态与结果登记在 [Seen-3 实验文档](CSGO_BENCHMARK_V2_EXP32_3MAPS.md)。

## 目标与归因

Stage 1 使用已经训练完成的 J8（`exp32_3maps_joint_constloc8`，50 epoch、5850 optimizer steps、恒定 αloc=8）；stage 2 从它的最终模型权重开始短程联合微调。将两阶段整体作为方法报告，同时分别记录 stage 1 的一次成本和每个 stage 2 分支的增量成本。复用 stage 1 可减少辅助消融的重复训练，但“总算力低于从头带辅助训练”须用实际 GPU 时间验证，不能仅凭新增步数断言。

设计中的三个主实验仅改变 stage 2 辅助组件。另配一个无辅助 S0 对照：三个主组对 J8@5850 的差异包含额外训练、optimizer 重置与 LR 重新启动的影响；只有与 S0 同节点比较，才能进一步判断辅助目标的贡献。设计要求 S0 匹配 optimizer 更新数、主任务曝光与 LR，不匹配辅助计算引入的 FLOPs/GPU 时间；本次实际曝光略不等，详见结果分析。

| ID | Experiment | Stage 1 权重 | Stage 2 perception | Stage 2 aux-loc | 用途 |
|---|---|---|---|---|---|
| S0_J8 | `exp32_3maps_stage2_control_j8` | J8@5850 | 关 | 关 | 等更新预算的续训对照 |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | J8@5850 | 开，最终 0.1 | 关 | perception 主消融 |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | J8@5850 | 关 | 开，最终 2 | aux-loc 主消融 |
| S3_J8 | `exp32_3maps_stage2_both_j8` | J8@5850 | 开，最终 0.1 | 开，最终 2 | 两者联合主消融 |

旧 J4 的 A0～A3 与从原始权重训练的 A1_J8～A3_J8 保持原配置、命令和输出目录。新组加载的是 J8 训练后权重，不再使用 `--joint-base` 动态复制学习率；J8 身份和初始化路径单独记录。

## 固定训练配方

| 项目 | Stage 2 设置 | 理由 |
|---|---|---|
| 初始化 | `outputs/csgo_1b/exp32_3maps_joint_constloc8/checkpoint-5850/model.safetensors`；仅加载模型权重 | 新建 optimizer、scheduler、Trainer/RNG 状态，local step 从 0 起 |
| 训练长度 | 显式 `max_steps=1800`，默认 `training_stop_after_step=1200` | 1200 是完整 1800 步调度的前缀，保留无缝继续的空间 |
| epoch 口径 | 按 J8 实测 117 steps/epoch，1200/1800 步约为 10.26/15.38 epoch | 以 optimizer steps 为准，不以估算 epoch 控制停止 |
| 主任务 | paired joint，αloc=8，源样本有效 batch=128 | 保留原任务、数据划分、GAS mean 归一化和每任务曝光 |
| 默认运行布局 | 单卡 microbatch=64、GAS=2、seed=42 | 其他布局须维持有效 batch；同一次 stage 2 恢复不改变布局 |
| 可训练模块 | 与 J8 相同；LoRA r=32、alpha=64、dropout=0.05，冻结 ViT | 不把容量或冻结策略混入组件消融 |
| LR 峰值 | J8 各组峰值 ×0.2：LLM/connector/生成 DiT/Action DiT LoRA、IO/time MLP、普通组为 2e-5；Action connector/norm/projector 为 1e-4 | 降低已适配表示的扰动，保留参数组间比例 |
| optimizer | AdamW（`adamw_torch`），β=(0.9,0.999)，eps=1e-8，weight_decay=0，max_grad_norm=1 | 显式固定已有 AdamW 常规设置；不加载 J8 optimizer 状态 |
| scheduler | 新建 cosine_with_min_lr；warmup_steps=60、warmup_ratio=0；min_lr_rate=0.1 | 保持各参数组 LR 比例，最终 LR 分别为 2e-6/1e-5 |
| 辅助启用 | 所启用辅助均在 local `[0,120]` 线性 ramp：perception `[0,0.1]`，aux-loc `[0,2]`，之后恒定 | 已有成熟的 stage 1；保留默认最终强度，让短 stage 2 有足够完整权重训练时间 |
| 保存 | 每 300 optimizer steps，最多保留 6 个完整 checkpoint，`save_only_model=False` | 保存模型、optimizer/DeepSpeed、scheduler、Trainer、RNG；支持跨作业恢复 |
| 默认停止 | local 1200 保存后退出；不会自动推理、评测或继续训练 | 人工查看 validation，再手动恢复到 1800 |

LR 相对 stage 1 峰值降低，但相对 stage 1 末期 LR 提高约 2 倍；这是温和重新 warmup，不是沿用 stage 1 的衰减尾段。Stage 2 首次训练绝不能完整 resume J8 的 optimizer/global_step；之后恢复 stage 2 自己的 checkpoint 则必须完整 resume，不能重新应用 J8 权重或重置调度。

runner 仍保留通用命令中的 `--num_train_epochs 50`，但新组显式传入的 `--max_steps 1800` 优先决定训练总长度；不会再训练 50 个 epoch。1800／60／300 等运行参数由 runner 固定，并保存在 `experiment_plan.json` 的 runtime 中；辅助调度和分组 LR 保存在 YAML 中。

保留 aux-loc 的 combined EM＋uncertainty、两个候选、共享定位噪声、`exp_sigma(lambda=5)` 与无 renorm；保留 perception 的 current_loc_head、非 EMA、vision-tower SmoothL1、detached GT attention 与 linear_1m_sigma。此轮不更换 teacher 或修改 loss 公式。

## 保存、验证与累计步数

| Stage 2 local step | 两阶段累计 step | 动作 |
|---:|---:|---|
| 0 | 5850 | 核验 J8 来源、模型结构和实际加载覆盖；既有 J8@5850 validation 作为初始化参照，不要求创建 checkpoint-0 |
| 300 | 6150 | 保存；必要时做异常诊断，不因辅助刚启用就淘汰 |
| 600 | 6450 | 保存；建议完整定位＋离散生成 validation |
| 900 | 6750 | 保存，供恢复或异常定位 |
| 1200 | 7050 | 保存并默认停止；四组预先指定的主要比较点，完整 validation |
| 1500 | 7350 | 仅恢复继续后保存 |
| 1800 | 7650 | 完整 stage 2 结束；S0 与入围主组同节点 validation，作为筛选后补充分析 |

命令的 `--checkpoint-step` 一律使用 **stage 2 local step**（600/1200/1800），不是累计 6450/7050/7650，也不是 stage 1 的 5850。保存目录以新实验名隔离。完整恢复只允许停止点变化，保持 max_steps=1800、LR、辅助调度、seed、数据后端和 batch 布局。

验证仍用固定三图 seen_validation、每图 500 条、相同 inference seed=42，逐图与等地图宏平均同时看 XY/Z/Pitch/Yaw、PSNR/SSIM/LPIPS/Boundary F1/FID。定位和生成必须来自同一个 checkpoint。local600 用于观察，local1200 保留四格完整主结果；需要继续时，S0 和参与 1800 结论的候选都继续同样预算。停止训练与验证独立执行，未设计自动单指标 patience。

不把单 seed 的小幅提升称为显著改善。当前基于 validation 的选择属于开发结果，配方和 checkpoint 冻结后才运行 test。仅变更 stage 2 seed 测到的是固定 J8 初始化下的随机性；最终若要验证完整两阶段方法稳健性，需配独立 stage 1 seed 和对应 stage 2/S0。

## 数据后端、初始化与实现验收

默认沿用 minimal asset manifest。切换全量时首次训练使用 `--asset-mode full --data-dir /path/to/preprocessed_data`（或 YAML 中 asset manifest=null），训练样本和 split 不变。恢复保留相同参数；推理/评测读取保存的 resolved_config。基础模型/pi05 路径可沿用 runner 的已有覆盖参数。

Stage 1 来源至少包含完整的 `checkpoint-5850/model.safetensors`、J8 的 `resolved_config.yaml` 与 `trainer_state.json`（优先 checkpoint 内，允许对应根目录最终状态）。首次执行检查 J8 身份、5850 完成状态、主配方/模型结构兼容和 safetensors 完整性，记录来源路径、文件 SHA-256、大小与 step。模型初始化时拒绝缺少可训练参数或 shape 不匹配；不能仅以文件存在视为成功加载。Stage 2 自身恢复依赖其保存的来源记录与完整 checkpoint，不应再次要求原始 J8 文件在场。

代码验收用 CPU 单元测试、配置检查和 train/infer/metrics dry run；检查旧组行为不变、新组 horizon/warmup/分组 LR 正确、初始化不继承旧 optimizer、自己的恢复不重置权重、两任务评测路由正确、数据后端切换与 provenance 一致。本次不启动任何实验、GPU batch、推理或 metric。

## 后续超参数消融与依据

先配置并比较上述四组，不额外展开网格。若 aux-loc 组相对 S0 明显损害生成，优先单因素将 αaux 从 2 降至 1；若问题集中在启用初期，再单独延长 ramp。若 S0 在内所有组都退化，先核对加载与 LR，再共同降低 stage 2 LR。这些后续分支只列为条件计划，待证据后再创建独立配置，不覆盖本轮。

分阶段适配和分组学习率可参考 [ULMFiT（ACL 2018）](https://aclanthology.org/P18-1031/)；继续适配本身会改变下游表现，可参考 [Don’t Stop Pretraining（ACL 2020）](https://aclanthology.org/2020.acl-main.740/)。它们支持设计原则，不证明本任务必然获益，也不决定这里 0.2 倍 LR、60/120/1800 步的具体取值。S0 与本项目的微调预算是结合当前训练状态提出的实验设计判断。

## 实现验收记录（2026-10-17）

已实现四份独立配置、runner 的 Stage 2 初始化／恢复入口、仅新组启用的严格模型加载检查，并补齐实验进度表、结果占位行与 record.md 命令。CPU 单测 29 项通过，24 份配置检查通过；四组的 train/infer/metrics dry run 和新增 Bash 命令块语法检查通过。旧 20 组的 YAML 字节、resolved 配方、启动命令、runtime 与恢复 fingerprint 保持不变，既有结果和用户对 record.md 的修改保留。

真实 J8 源文件通过只读预检：4,577,302,678 bytes、2830 tensors（1128 LoRA），SHA-256 为 `d970e75d73c65e0fe7751d703abf94683fd7a50aca3afab731d756dd633ebbea`。实际安装的 Transformers scheduler 经 CPU 检查，在 local60 达到 2e-5／1e-4，在 local1800 达到 2e-6／1e-5，保持 5 倍组间比例。完整大模型的实际加载覆盖检查将在首次训练启动时执行；本次未构建大模型或运行训练、推理、评测，不据此声称指标改善。

## 2026-10-23：新增九组 Stage 2 消融设计

本节是 S0_J8～S3_J8 已完成实验之后的新设计，前述四组的历史 `save_steps=300`、命令与结果保持原记录。以下九组分别占用新运行名和输出目录；前七组先运行并由人工查看完整 validation，之后才手动决定执行后两组。没有自动实验队列，也不根据评测指标自动触发后两组。2026-10-23 九组加既有 24 组当时共 33 个独立运行名；增加 U5／P5 后当前计划共 35 个。**2026-10-28 修订 B1／P4／U4 三行尚无本地训练产物的配方**：B1 联合 P3 的 perception 0.05 ramp 与 U2 的 aux-loc 1 ramp，保留 `attention=true`、`candidate_tau=1`；P4 仅把 P3 的 attention 改为 false；U4 仅把 U2 的 candidate_tau 改为 0.001。三者运行名中的 S3／S1／S2 只标识既定运行，不表示沿用原高权重。

| 批次 | ID／运行名 | 配置参考 | 本组变化 | perception 权重 | aux-loc 权重 |
|---|---|---|---|---|---|
| 先七组 | P1_S1_J8 `exp32_3maps_P1_S1_J8` | S1_J8 | 去除 perception ramp | 从 local step 0 恒定 0.1 | 关 |
| 先七组 | U1_S2_J8 `exp32_3maps_U1_S2_J8` | S2_J8 | 去除 aux-loc ramp | 关 | 从 local step 0 恒定 2 |
| 先七组 | B1_S3_J8 `exp32_3maps_B1_S3_J8` | P3_S1_J8＋U2_S2_J8 | 联合两项低权重 ramp；attention=true、candidate_tau=1 | local `[0,120] → [0,0.05]`，之后恒定 | local `[0,120] → [0,1]`，之后恒定 |
| 先七组 | P2_S1_J8 `exp32_3maps_P2_S1_J8` | S1_J8 | perception 终值改为 0.2 | local `[0,120] → [0,0.2]`，之后恒定 | 关 |
| 先七组 | P3_S1_J8 `exp32_3maps_P3_S1_J8` | S1_J8 | perception 终值改为 0.05 | local `[0,120] → [0,0.05]`，之后恒定 | 关 |
| 先七组 | U2_S2_J8 `exp32_3maps_U2_S2_J8` | S2_J8 | aux-loc 终值改为 1 | 关 | local `[0,120] → [0,1]`，之后恒定 |
| 先七组 | U3_S2_J8 `exp32_3maps_U3_S2_J8` | S2_J8 | aux-loc 终值改为 4 | 关 | local `[0,120] → [0,4]`，之后恒定 |
| 后两组 | P4_S1_J8 `exp32_3maps_P4_S1_J8` | P3_S1_J8 | 完整沿用 P3，仅 perception `attention=false` | local `[0,120] → [0,0.05]`，之后恒定 | 关 |
| 后两组 | U4_S2_J8 `exp32_3maps_U4_S2_J8` | U2_S2_J8 | 完整沿用 U2，仅 aux-loc `candidate_tau=0.001` | 关 | local `[0,120] → [0,1]`，之后恒定 |

表中的 S1／S2／S3、P3／U2 都只是**配方参考**，从未作为这些新组的模型权重来源。九组首次训练统一只加载 J8@5850 `model.safetensors`，从 local step 0 新建 optimizer、scheduler、Trainer/RNG 状态；沿用 paired joint、αloc=8、seed=42、原分组 LR（普通组峰值 2e-5，Action connector/norm/projector 峰值 1e-4）、AdamW、60 步 warmup、cosine min LR 及其他未注明参数。每组默认完整训练 `max_steps=1800`，`--stop-after-step 0`，每 600 local steps 保存完整 checkpoint（600／1200／1800）。单卡命令默认 microbatch=4，GAS 按有效 batch 128 自动设置；更换 GPU 布局时先核对有效 batch，恢复时保持原布局。B1 与 S0／P3／U2 形成四格联合比较，不从单项 loss 预设收益相加；P4 直接对照 P3，U4 直接对照 U2。

九组只在 local 1200 和 1800 做 seen_validation 三图的定位与离散生成，不评测 600。local 1800 的 test 同一 checkpoint 执行定位、离散生成和连续生成；`infer --task all --split test` 自动路由三项推理，`metrics` 复用定位输出并生成离散、连续各自的三图汇总。test 定位／离散每图 2,000 条（共 6,000），连续每图 20×64=1,280 帧（共 3,840），连续指标沿用 clip_length=16、stride=16、FVD size=224。测试结果只报告，不用于选择后两组配置或决定是否启动。所有 `--checkpoint-step` 使用 Stage 2 local step。默认 minimal 资源；首次改用 full 时加 `--asset-mode full --data-dir /path/to/preprocessed_data`，之后读取本组保存的 `resolved_config.yaml`，从本组 checkpoint 恢复时保持相同后端与路径，并用 `--resume --stop-after-step 0` 保持完整 1800 步调度。逐组可复制命令和结果记录见 [record.md](record.md) 与 [Seen-3 实验文档](CSGO_BENCHMARK_V2_EXP32_3MAPS.md)。

## 2026-10-26：九组消融本地状态

九组中 P1_S1_J8、P2_S1_J8、U2_S2_J8 已完成 local1800，local1200／1800 的定位与离散生成 validation、local1800 的定位／离散／连续 test 已同步；U1 仅有 600／1200 checkpoint 和从 1200 恢复的计划，P3 仅有 600 checkpoint，U3 仅有配置／计划，B1／P4／U4 本地无训练输出。其余组的远端状态不据本地缺项推断。数值、源 JSON 与进度证据见[主实验文档九组结果](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#nine-ablation-results-20261026)及[2026-10-26 分析](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#stage2-new-results-20261026)。

## 2026-10-27：U1／P3 同步状态

U1_S2_J8 与 P3_S1_J8 已同步至本地 local1800/1800；两组均有 local600／1200／1800 checkpoint、local1200／1800 定位及离散生成 validation，以及 local1800 定位／离散／连续 test。U1 从本组 local1200、P3 从本组 local600 恢复，首次初始化均为 J8@5850 模型权重。截至 2026-10-27 九组中五组完成；指标与源文件见[主实验文档九组结果](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#nine-ablation-results-20261026)及[test结果总表](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#test-overall-20261027)。

<a id="u5-aux05-20261027"></a>

## 2026-10-27：U5_S2_J8 aux-loc 终值 0.5 独立补充

2026-10-23 的九组加既有 24 组在当时共 33 个独立运行名；本次在九组之外新增 `exp32_3maps_U5_S2_J8`，计划总数为 34。此次配置登记仅新增 U5 的[静态配置](csgo_configs/exp32_3maps_U5_S2_J8.yaml)和[手动命令](record.md#exp32_3maps-u5_s2_j8-2026-10-27-手动命令）；未由此次配置操作启动训练或评测。没有配置 aux-loc 终值 0.25 的分支。

U5 完整继承 U2_S2_J8：仅把 aux-loc 权重从 local `[0,120] → [0,1]` 改为 `[0,120] → [0,0.5]`，之后恒定 0.5。perception 关闭；paired joint、αloc=8、普通组峰值 LR 2e-5、Action connector/norm/projector 峰值 1e-4、AdamW、cosine min LR、warmup 60、max_steps 1800、seed 42、单卡 microbatch 4 和目标有效 batch 128 均不变。首次训练只加载 J8@5850 `model.safetensors` 的模型权重，新建 optimizer、scheduler、Trainer/RNG 并从 local step 0 开始；U2 是配方来源，不是权重来源。

U5 在 local 600／1200／1800 保存完整 checkpoint；local 1200／1800 分别做 seen_validation 定位与离散生成；local1800 test 运行定位、离散生成和连续生成。默认 minimal 资源。首次改用 full 时在 train 命令加入 `--asset-mode full --data-dir /path/to/preprocessed_data`，后续 infer/metrics 使用本组 `resolved_config.yaml`；若中断，只从 U5 自己的完整 checkpoint 使用 `--resume --stop-after-step 0` 恢复，并保持原 GPU／microbatch 布局、数据后端与路径。所有步骤均需人工启动；此处保留 2026-10-27 的配置登记快照。2026-10-29 的完成进度和结果见[主实验文档 U5 补充](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#u5-aux05-20261027)。

<a id="p5-perception025-20261028"></a>

## 2026-10-28：P5_S1_J8 perception 终值 0.025 独立补充

在原九组与 U5 之外，新增 `exp32_3maps_P5_S1_J8`；计划总数由 34 增至 35（原九组＋U5＋P5 共 11 个新增独立运行名）。P3_S1_J8 是配方参考，只把 perception 权重从 local `[0,120] → [0,0.05]` 改为 `[0,120] → [0,0.025]`，之后恒定 0.025；ramp 保留，aux-loc 关闭。P1 去除 ramp，因此不是本组的直接配方对照。本次只增加 0.025 一个低权重点，不新增 0.0125；比较 P3/P5 可观察低于 0.05 的区间，S1 的 0.1 提供较高权重参照，S0 提供相同 Stage 2 更新预算的无辅助参照。以同节点 validation 判断降低权重的收益与代价，不预设权重越小越好。

P5 首次训练只加载 J8@5850 `checkpoint-5850/model.safetensors` 模型权重，从 local step 0 新建 optimizer、scheduler、Trainer/RNG；不从 P3@1800 续训。保留 paired joint、αloc=8、普通组峰值 LR 2e-5、Action connector/norm/projector 峰值 1e-4、AdamW、cosine_with_min_lr（`min_lr_rate=0.1`）、warmup 60、max_steps 1800、seed 42、单卡 microbatch 4 和目标有效 batch 128。local 600／1200／1800 保存完整 checkpoint。默认 minimal 资源；首次改用 full 时在 train 命令加入 `--asset-mode full --data-dir /path/to/preprocessed_data`，后续 infer/metrics 使用本组保存的 `resolved_config.yaml`。中断后仅从 P5 自己的完整 checkpoint 加 `--resume --stop-after-step 0` 恢复，保持原 GPU／microbatch 布局、数据后端与路径。

local 1200／1800 使用 seen_validation 三图、seed 42，每图 500 条，定位与离散生成必须来自同一 checkpoint；以 local1800 validation 作为 P5 与 P3/S1/S0 的主要比较，local1200 观察走势，不拼接不同 checkpoint 的最佳指标。local 1800 的 test 使用 seen_discrete_test 定位与离散生成（每图 2,000 条），以及 seen_continuous 连续生成（每图 20×64 帧），仅用于报告，不参与选型。`--checkpoint-step` 均指 Stage 2 local step。此次配置登记仅新增[静态配置](csgo_configs/exp32_3maps_P5_S1_J8.yaml)、[手动命令](record.md#exp32_3maps-p5_s1_j8-2026-10-28-手动命令)与[当时待填的主实验文档结果表](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#p5-perception025-20261028)；未由此次配置操作启动训练或评测。

## 2026-10-29：P/U 消融结果同步

原九组中 P1／P2／P3／P4／U1／U2／U3 共 7/9 组已在本地完成 local1800，两节点 seen_validation 定位／离散生成和 local1800 test 三任务齐全；B1／U4 尚无 1200／1800 评测；本次早期曾见 local600，最终复核时 B1 仅剩计划／配置、U4 训练目录不可见，不能将其登记为当前可用 checkpoint。九组外 U5／P5 也完成，P/U 共 9/10 组完整。完整组当前本地模型 checkpoint 为 local1200／1800；设计中的 local600 保存节点不代表现在仍已同步。数值、来源与当前状态见[主文档九组结果](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#nine-ablation-results-20261026)、[U5 补充](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#u5-aux05-20261027)、[P5 补充](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#p5-perception025-20261028)和[新分析](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#stage2-pu-results-20261029)。

<a id="single-stage2-controls-20261030"></a>

## 2026-10-30：G0／L0 单任务 Stage 2 无辅助续训设计

在此前 35 个独立运行名之外新增 S0_G0=`exp32_3maps_stage2_control_g0`、S0_L0=`exp32_3maps_stage2_control_l0`，总计划数为 37。它们是单任务的额外更新对照，与历史 S0_J8～S3_J8 四组、九组消融及 U5／P5 各占独立输出目录。G0@5850 和 L0@5850 的来源模型 checkpoint 已确认存在；新组的严格模型加载和实际训练尚未执行。执行模板见 [record.md](record.md#single-stage2-controls-20261030)，进度及指标占位见[主实验文档](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#single-stage2-controls-20261030)。

| ID | 唯一 Stage 1 权重来源 | 任务采样 | 继承架构／冻结 | 有效验证与 test |
|---|---|---|---|---|
| S0_G0 | `outputs/csgo_1b/exp32_3maps_gen_single/checkpoint-5850/model.safetensors` | `gen_single`，`task_mix_ratio=0`（仅生成），`is_multi_task_balanced=false` | 保留 G0 的共享及生成 LoRA、生成模块与冻结设置；ViT、非活动定位头冻结 | local1200／1800 生成 validation；local1800 离散＋连续生成 test |
| S0_L0 | `outputs/csgo_1b/exp32_3maps_loc_single/checkpoint-5850/model.safetensors` | `loc_single`，`task_mix_ratio=1`（仅定位），`is_multi_task_balanced=false` | 保留 L0 的共享及定位 LoRA、定位模块与冻结设置；ViT、非活动生成头冻结 | local1200／1800 定位 validation；local1800 定位 test |

两组首次只严格加载各自来源的 **模型权重**，要求模型结构、可训练参数与 shape 覆盖检查通过；各自新建 optimizer、scheduler、Trainer/RNG，从 Stage 2 local step 0 开始。不得继承 source@5850 的 optimizer／scheduler，不能把 G0 与 L0 的源模型互换。保持 source 的单任务架构与 LoRA／冻结语义；αloc=1（生成组不参与生成 loss），perception、aux-loc 等全部辅助 loss 关闭。ViT 始终冻结，非活动任务 head 冻结。两组 seed=42、源样本目标有效 batch=128，单 GPU 默认 microbatch=4、GAS=32；实际任务曝光需以 Trainer 记录核对尾批。

两组均完整训练 local1800，不设置 local1200 自动停止。共享参数、活动 LoRA 与 IO 的 peak LR 为 2e-5；仅 S0_L0 中活动的 Action connector／norm／projector 为 1e-4。新 AdamW β=(0.9,0.999)、eps=1e-8、weight decay=0、gradient clip norm=1；`cosine_with_min_lr`、warmup=60、`min_lr_rate=0.1`。local600／1200／1800 保存完整 checkpoint，`save_total_limit=6`；local1200／1800 只评活动任务的 seen_validation（固定三图、每图 500、seed=42），local1800 对 seen_discrete_test 和 seen_continuous 只评有效 test 任务。生成的 test 包含离散与连续，定位的 test 仅定位。CLI `--task all` 应由单任务配置正确路由，不把非活动任务 N/A 视为缺失。所有 `--checkpoint-step` 是本组 local step，累计步数仅作解释：7050／7650。

默认 minimal asset；首次改用 full 时 train 追加 `--asset-mode full --data-dir /path/to/preprocessed_data`。本组后续 infer/metrics 读取保存的 `resolved_config.yaml`，不能中途切换后端或路径。中断后只从同组完整 checkpoint 加 `--resume --stop-after-step 0`，保持相同 GPU 数、microbatch、有效 batch 和数据后端／路径；可改 GPU 编号。首次训练不传 `--resume`。命令是手动模板，不构成实验自动队列。

比较目标是把单任务 source@5850 与同任务 local1200／1800 放在对应任务指标上，看继续更新后的变化，再与 J8@5850→S0_J8 这条完整联合两阶段路径并列分析。可设计匹配 Stage 2 更新数、每任务源样本目标曝光和续训 schedule；**总算力不同**，且 G0／L0 使用各自不同的 Stage 1 权重，不能把跨模型差值解释为单一组件效应。辅助 loss 的同源增量比较仍然使用 S0_J8。当前只有配置计划，没有新组 validation／test 数值，不推断性能。

首次来源同步只要求源 `model.safetensors`、源 `resolved_config.yaml` 和完成 5850 的 `trainer_state.json`（优先 checkpoint 内，可回退根目录最终状态）；不需要源 optimizer／scheduler／RNG。首次预检区分 G0／L0／J8 的任务和 LoRA 结构并记录 SHA-256。本组中断后的 `--resume` 才要求本组完整 checkpoint，恢复不再要求来源模型仍在场。2026-10-30 配置、路由及来源只读预检已通过，验收记录见[主实验文档](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#single-stage2-controls-20261030)。

<a id="b2-stage2-j8-20261030"></a>

## 2026-10-30：B2_S3_J8=P4＋U2 的联合补充设计

原 37 个计划运行名之外新增 `exp32_3maps_B2_S3_J8`，当前计划总数为 **38**。这是独立的 Seen-3 Stage 2 运行，旧九组、U5／P5 与单任务 S0_G0／S0_L0 的历史矩阵、数字和结果保持其日期口径。B2 尚未训练或评测；[主实验文档的独立进度与指标占位](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#b2-stage2-j8-20261030)不代表实测结果。

| 维度 | B2 固定设置 | 可归因比较 |
|---|---|---|
| 初始化 | 仅从 J8@5850 `outputs/csgo_1b/exp32_3maps_joint_constloc8/checkpoint-5850/model.safetensors` 严格加载模型权重；重建 optimizer／scheduler／Trainer/RNG，local step=0 | P4／U2／B1 都不是权重来源 |
| 联合主任务 | paired joint、αloc=8、原 J8 联合分支架构／LoRA／冻结／采样设置；三图 ancient、dust2、nuke | S0_J8 同源无辅助对照 |
| perception | P4 配方，perception 权重 local `[0,120]→[0,0.05]`，之后恒定；`attention=false`；其余 perception 分支设置沿用 P4 | B1 唯一差别是 `attention=true` |
| aux-loc | U2 配方，combined 权重 local `[0,120]→[0,1]`，之后恒定；`aux_loc_combined_candidate_tau=1`、`aux_loc_combined_num_samples=2`；其余 aux-loc 分支设置沿用 U2 | 与 P4 单辅助比较联合效应 |
| 优化 | seed=42；源样本有效 batch=128，单 GPU microbatch=4、GAS=32；普通组 peak LR 2e-5，Action connector／norm／projector peak LR 1e-4；AdamW β=(0.9,0.999)、eps=1e-8、WD=0、clip norm=1；cosine_with_min_lr、warmup 60、min_lr_rate=0.1 | S0／P4／U2／B2 同 local 步数 |
| 节点 | local1800 完整训练；600／1200／1800 保存，最多 6 份；1200／1800 validation 定位＋离散，1800 test 定位＋离散＋连续 | 不按 test 结果选配方 |

默认 minimal 资产；首次 train 选 full 时加 `--asset-mode full --data-dir /path/to/preprocessed_data`，infer/metrics 读取 B2 自身 `resolved_config.yaml`。首次 train 不加 `--resume`；只有 B2 自身完整 checkpoint 的中断恢复才加 `--resume --stop-after-step 0`，保持原 GPU 数、microbatch、有效 batch 与数据后端／路径。local1200／1800 分别对应两阶段累计 7050／7650 步，但 CLI `--checkpoint-step` 一律用 local step。执行模板见 [record.md](record.md#exp32_3maps-b2_s3_j8-2026-10-30-手动命令)。

以同 checkpoint、同 split 的验证指标比较 S0／P4／U2／B2 四格；B1 对 B2 单独检验 attention 开关。测试只报告：seen_discrete_test 定位和离散每图 2,000 条，seen_continuous 每图 20×64 帧。Seen-10 `exp37_s2_both` 共享 P4＋U2 辅助配方，但使用 Seen-10、ramp 400、local6000；本组 Seen-3 的 ramp120、local1800 与其不可混为同口径实验。
