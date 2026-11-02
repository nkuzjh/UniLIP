# CSGO Benchmark v2：exp37 Seen-10 两阶段实验协议

状态：**配置与命令待执行**。本文登记六个新的训练运行：`exp37`、`exp37_s2_gen`、`exp37_s2_loc`、`exp37_s2`、`exp37_s2_both`、`exp37_s2_both_b1`（1 次 Stage 1 + 5 次 Stage 2）。目前不报告训练完成、checkpoint、参数量或指标。运行入口为 [`scripts/run_exp37_seen10.py`](scripts/run_exp37_seen10.py)，手动命令见 [`record.md`](record.md)。本协议只覆盖 Benchmark v2 manifest 中完整十张 Seen 地图，不是三图消融。

## 研究问题与比较边界

`exp37` 从原始 UniLIP 与 π0.5 权重全新训练 Seen-10 联合基线 J8，定位权重 αloc=8，不启用辅助损失。其完成的 Stage 1 最终模型权重分别初始化 `exp37_s2`（无辅助）、`exp37_s2_both`（P4 perception + U2 aux-loc）和 `exp37_s2_both_b1`（P3 perception + U2 aux-loc）。`exp37_s2_gen` 与 `exp37_s2_loc` 则分别只从旧 Seen-10 `exp32_gen@19500`、`exp32_loc@19500` 的模型权重开始。**六个名字代表六次新训练**；Stage 2 的五次训练各有独立 optimizer、scheduler、Trainer local step 和 RNG。

| 新实验 | 训练路线 | 首次初始化来源 | 辅助损失 | 评测任务 |
|---|---|---|---|---|
| `exp37` | Stage 1，联合 J8 | 原始 UniLIP + π0.5 | 均关闭 | 定位、离散生成、连续生成 |
| `exp37_s2_gen` | Stage 2，生成单任务 | 旧 `exp32_gen@19500` 模型 | 均关闭 | 离散生成、连续生成 |
| `exp37_s2_loc` | Stage 2，定位单任务 | 旧 `exp32_loc@19500` 模型 | 均关闭 | 定位 |
| `exp37_s2` | Stage 2，联合对照 | 新 `exp37` 完整训练最终模型 | 均关闭 | 定位、离散生成、连续生成 |
| `exp37_s2_both` | Stage 2，联合辅助 | 同一新 `exp37` 完整训练最终模型 | P4 + U2 | 定位、离散生成、连续生成 |
| `exp37_s2_both_b1` | Stage 2，联合辅助，独立 B1 候选 | 同一新 `exp37` 完整训练最终模型 | P3 + U2 | 定位、离散生成、连续生成 |

Stage 1 是新增训练预算，联合两阶段总训练量应作为预算报告，不应称作与旧单阶段 baseline **FLOPs 相等**。五个 Stage 2 都固定本地 `max_steps=6000`，终点 local 6000 是表格比较 checkpoint；Stage 1 最终 checkpoint 只作为参照。联合三组必须来自同一个实际完成的 `exp37` 终点，由 runner 动态解析实际 step 并固定来源身份，不猜定终点为 19500，也不在 Stage 1 未完成时自动排队或退回较早 checkpoint。

## 训练配置

| 项目 | Stage 1 `exp37` | 五个 Stage 2 |
|---|---|---|
| 数据与 epoch/step | Seen-10 manifest 全十图；50 epochs，实际最终 step 动态确定 | Seen-10 全十图；local `max_steps=6000`，由旧 15k→50k 的曝光比例 1800→6000 推得 |
| 任务与权重 | joint，αloc=8，无辅助 | joint 三组 αloc=8；gen-only 配置 αloc=1 但无定位任务而不参与损失；loc-only αloc=1 且作用于定位损失 |
| 有效 batch | 目标源样本 batch=128；默认单卡 microbatch=4、GAS=32 | 同左；改变卡数时须保持目标有效 batch 与恢复布局 |
| 普通可训练参数峰值 LR | `1e-4` | `2e-5` |
| Action connector / norm / projector 峰值 LR | `5e-4` | `1e-4`，仅活动任务对应私有头参与 |
| 优化器 | AdamW，β=(0.9, 0.999)，eps=`1e-8`，weight decay=0，clip norm=1 | 相同，重新创建 |
| 学习率 | 保留 J8 的 `cosine_with_min_lr`、`min_lr=1e-5`；`warmup_ratio=0.003` | cosine with `min_lr_rate=0.1`；warmup 200 local steps |
| 固定随机种子 | 42 | 42；首次来源加载后建立本组 RNG |
| checkpoint | 每 4000 step 保存，最多保留 6 个完整状态；50 epochs 真正最终状态也保存 | 每 2000 local step 保存，最多保留 6 个完整状态；不在中途自动停止 |
| validation / test | validation @8000 与实际 final；test @final | validation @4000、6000；test @6000 |

Stage 2 首次启动**严格只加载来源 checkpoint 的模型权重**，不继承来源 optimizer、scheduler、Trainer local step 或 RNG。`--resume` 只用于从**本实验自己的完整 Stage 2 checkpoint**恢复，同一实验保持设备布局、microbatch、资产后端和数据路径，不重新应用来源初始化。`--stop-after-step 0` 表示完整跑到既定终点，不能误作零步训练或旧版 2400 提前停机。生成单任务只训练自己的生成头与相应 LoRA；定位单任务只训练自己的定位头与相应 LoRA，非活动头冻结。

这里的 Stage 1 `min_lr=1e-5` 是原 J8 的 scheduler 参数。已核对当前 Transformers 实现：它先除以 optimizer 的默认 LR 得到比例 0.1，再应用于各参数组，因此普通组终点为 `1e-5`，Action 私有组为 `5e-5`；不是让所有组分别降到同一个绝对值。保留原配方，不引入新的 scheduler。Stage 2 明确使用比例 0.1，对应普通组终点 `2e-6`、Action 私有组终点 `1e-5`。

Stage 1 最终步数按 Trainer 实际完成状态解析；当前 `DefaultFlowCallback` 会在 `state.max_steps` 额外保存，即使该步不是 4000 的整数倍，也无需预估或硬编码最终步数。

### Stage 2 辅助损失的唯一差异

`exp37_s2` 是无辅助对照。`exp37_s2_both` 在相同 Stage 1 终点模型上，加入 P4 与 U2：

- P4 perception：local 0→400 将权重线性从 0 升至 0.05，随后保持 0.05；`loc_perception_use_attention_weight=false`，使用当前定位头、teacher 特征 `detach`、无 EMA；vision tower token SmoothL1，und tokens 使用 `linear_1m_sigma`。
- U2 aux-loc：local 0→400 将权重线性从 0 升至 1，随后保持 1；`combined`、每样本 2 个候选、`candidate_tau=1`、`unc_metric=residual_l1_normed`、`unc_tau=1`、`unc_min_weight=0.05`、定位共享噪声、`exp_sigma` 的 λ=5；不做额外 loss renormalization。

旧三图 B1 是 **P3 + U2、`attention=true`**；`exp37_s2_both` 是 **P4 + U2、`attention=false`**。新增 `exp37_s2_both_b1` 在同一 Seen-10 Stage 2 配方下仅将 perception 的 `loc_perception_use_attention_weight` 从 `false` 改为 `true`，其余 ramp、U2、αloc=8、LR、warmup、优化器、local 6000 与评测节点均与 `exp37_s2_both` 一致。新增组有独立实验身份和训练状态，不替换 P4+U2 组；旧三图 B1 的地图、阶段预算与来源均不同，不构成 Seen-10 已证实增益的证据。

## 来源文件与恢复

首次启动单任务 Stage 2 时，同步各自旧 `checkpoint-19500/` 中的 `model.safetensors`、`config.json`、`trainer_state.json` 和 `training_args.bin` 即可，不需要源 optimizer。runner 结合这些实际文件与仓库中原 `exp32_gen/loc.yaml` 核对：仓库 YAML 表示原计划配方，checkpoint 元数据、任务日志和张量头是实际运行证据；不伪造旧 `resolved_config.yaml`。外部训练日志及单独导出的 adapter metadata 不作为必需文件。

三个联合 Stage 2 首次启动需要新 `exp37` 的 `resolved_config.yaml`、`experiment_plan.json`，以及实际完成 checkpoint 的 `model.safetensors`、`trainer_state.json`；runner 固定来源路径和 SHA-256，并核对三组使用同一来源。所有 Stage 2 的 `--resume` 均读取自己的计划、配置和完整 checkpoint，不再要求源模型文件仍在场；本组 optimizer/scheduler 与全部进程对应的 RNG 状态必须齐全。

## 数据、评测和结果记录

训练、infer、metrics 均使用同一资产后端。默认 `--asset-mode minimal`，资产由 `data/csgo_benchmark_v2/minimal_dataset_report.json` 映射；首次运行须确认 manifest 和十图资产可达。需要使用全量资产时，仅在**首次 train** 改为 `--asset-mode full --data-dir /path/to/preprocessed_data`，对应推理与 metrics 从本组保存的 `resolved_config.yaml` 读取同一后端与路径。恢复训练也须保持首次运行的后端和数据路径；禁止将 minimal 与 full 产物混算。`record.md` 中可执行命令明确使用 minimal 后端。

validation 每张地图定位与离散生成各 500 条，总计各 5,000；continuous generation 仅在 test 运行。test 每张地图定位与离散生成各 2,000 条，总计各 20,000；连续生成每图 1,280 帧，总计 12,800。推理设置、seed、clip length、stride、FVD 尺寸应从同一已解析配置读取；指标按十张地图等权宏平均，同时留存逐图明细。`--task all` 按实验路线分发：joint 三任务，gen-only 仅离散与连续，loc-only 仅定位。定位指标由 infer 生成；生成指标由 metrics 计算。记录中仍按统一流程调用两命令；loc-only 的 metrics 不重复计算定位，正式定位摘要应核对 infer 产物。

| 实验 | validation | 固定正式 test checkpoint | 当前结果 |
|---|---|---|---|
| `exp37` | 8000、实际 final | Stage 1 实际 final，仅参照 | 待执行 |
| `exp37_s2_gen` | local 4000、6000 | local 6000 | 待执行 |
| `exp37_s2_loc` | local 4000、6000 | local 6000 | 待执行 |
| `exp37_s2` | local 4000、6000 | local 6000 | 待执行 |
| `exp37_s2_both` | local 4000、6000 | local 6000 | 待执行 |
| `exp37_s2_both_b1` | local 4000、6000 | local 6000 | 待执行 |

结果写入 [`csgo_benchmark_v2_experiments_results.md`](csgo_benchmark_v2_experiments_results.md) 的 exp37 专区；Table 1 身份映射见 [`csgo_benchmark_v2_table1_seen10.md`](csgo_benchmark_v2_table1_seen10.md)。只有实际完成并核对 checkpoint、十图覆盖、后端、任务路由及宏平均后，才填指标和参数量。历史表格数值不因新增实验而改变。

## 实现验收（2026-10-30 五组快照）

以下验收仅覆盖当日原五组，不适用于 2026-11-01 新增的 `exp37_s2_both_b1`。

- `python scripts/run_exp37_seen10.py validate` 和脚本编译通过；新增 runner 的 15 项 CPU 测试通过，覆盖来源、动态 final、任务路由、完整十图、恢复状态和执行前拒绝错误来源。
- 原 `run_exp32_3maps` 的 41 项回归测试通过。五组实验在 minimal/full 两种后端的 train、infer、metrics 共 30 个 dry-run 计划通过，无新训练或评测输出目录。
- `record.md` 五组顺序执行命令及恢复模板共 6 个代码块通过 `bash -n`；记录文件保留 CRLF。`git diff --check` 通过。
- 已对实际旧 `exp32_gen/loc@19500` 完成状态、训练参数、任务日志、张量头和权重 SHA-256 做只读核验；无须旧 `resolved_config.yaml`。新 `exp37` 尚未训练，其来源完成性将在联合 Stage 2 首次执行前检查。
- 未启动训练、模型推理或指标计算，未做真实新模型加载；严格全模型 key/shape 加载校验将在用户首次训练时执行。尚无本组效果结论。

## 新增 B1 候选验收（2026-11-01）

- 新 YAML 与原 `exp37_s2_both.yaml` 的解析差异仅为实验身份和 `loc_perception_use_attention_weight: false→true`，P3＋U2 的终值0.05／1、ramp400、αloc=8 均已核对。原五份 YAML 字节未变。
- 六组配置校验与17项 CPU 测试通过，覆盖新组路由、固定配方、评测节点、三个联合组的来源一致性，以及删除 Stage1 源文件后从本组 checkpoint 恢复的规划。两个 peer 同时存在、前者匹配而后者 path/hash 不符的情况也会拒绝。
- 原五组在 minimal/full 两种后端的62个训练与评测计划、配置、runtime 和 fingerprint 与变更前一致；新组14个计划覆盖训练、4000／6000 validation、6000 test，检查通过。这里的 dry-run 只生成计划，不加载模型或计算指标。
- `record.md` 新增的首次运行与中断恢复两段命令通过 `bash -n`，保留 CRLF；`git diff --check` 通过。训练先运行至 local6000，再按命令顺序评测4000／6000 validation及6000 test。
- 未创建 exp37 系列训练或评测输出目录，未启动训练、推理或评测。实际模型加载与 Trainer 恢复执行未在此次 CPU 验收中验证；训练、validation、test 和参数审计仍待执行。
