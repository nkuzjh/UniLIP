# J8 两阶段辅助微调设计

设计日期：2026-10-17。本文件先于本次代码和配置变更记录方案；实验尚未启动。执行命令集中放在 [record.md](record.md)，状态与结果登记在 [Seen-3 实验文档](CSGO_BENCHMARK_V2_EXP32_3MAPS.md)。

## 目标与归因

Stage 1 使用已经训练完成的 J8（`exp32_3maps_joint_constloc8`，50 epoch、5850 optimizer steps、恒定 αloc=8）；stage 2 从它的最终模型权重开始短程联合微调。将两阶段整体作为方法报告，同时分别记录 stage 1 的一次成本和每个 stage 2 分支的增量成本。复用 stage 1 可减少辅助消融的重复训练，但“总算力低于从头带辅助训练”须用实际 GPU 时间验证，不能仅凭新增步数断言。

用户要求的三个主实验仅改变 stage 2 辅助组件。另配一个无辅助 S0 对照：三个主组对 J8@5850 的差异包含额外训练、optimizer 重置与 LR 重新启动的影响；只有与 S0 同节点比较，才能进一步判断辅助目标的贡献。S0 匹配 optimizer 更新数、主任务曝光与 LR，不匹配辅助计算引入的 FLOPs/GPU 时间。

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
