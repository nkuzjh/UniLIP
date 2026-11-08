# CSGO Benchmark v2：exp37 Seen-10 两阶段实验协议

后续接口（2026-11-03）：五个Stage2的实际完成 `checkpoint-6000` 将作为
[Unseen-4每图独立few-shot](CSGO_BENCHMARK_V2_EXP37_UNSEEN4.md)的来源。
适配使用独立runner和400步新optimizer，不改变本文Seen训练配方或恢复方式。
两个both候选都保留配置；Unseen query不得用于选择最终候选或调参。

状态（2026-11-08）：本文登记六个新的训练运行：`exp37`、`exp37_s2_gen`、`exp37_s2_loc`、`exp37_s2`、`exp37_s2_both`、`exp37_s2_both_b1`（1 次 Stage 1 + 5 次 Stage 2）。前四组已有完整 checkpoint 和原batch1的validation/test评测；`exp37_s2_both` 已完成 local6000/6000，最终权重与状态及batch8三项test均已同步。最新batch8队列完成了`exp37_s2`定位／离散／连续推理，但在指标前置校验时报错退出：定位结果有效，离散／连续图片与推理清单齐全、未执行指标计算；`exp37_s2_loc`和`exp37_s2_gen`未由该队列执行，本机没有这两组的batch8结果。错误原因为定位摘要中的清单路径与校验期望路径字符串不同，尽管二者解析到同一文件；细节见[本页2026-11-08根因记录](#exp37-bs8-queue-root-cause-20261108)。validation状态不变：both仅有local4000定位（batch=1）；该节点离散生成中断，local6000的validation尚未同步。`exp37_s2_both_b1` 本地仍无产物，远端状态待确认。运行入口为 [`scripts/run_exp37_seen10.py`](scripts/run_exp37_seen10.py)，手动命令见 [`record.md`](record.md)。本协议只覆盖 Benchmark v2 manifest 中完整十张 Seen 地图，不是三图消融。

## 研究问题与比较边界

`exp37` 从原始 UniLIP 与 π0.5 权重全新训练 Seen-10 联合基线 J8，定位权重 αloc=8，不启用辅助损失。其完成的 Stage 1 最终模型权重分别初始化 `exp37_s2`（无辅助）、`exp37_s2_both`（P4 perception + U2 aux-loc）和 `exp37_s2_both_b1`（P3 perception + U2 aux-loc）。`exp37_s2_gen` 与 `exp37_s2_loc` 则分别只从旧 Seen-10 `exp32_gen@19500`、`exp32_loc@19500` 的模型权重开始。**六个名字代表六次新训练**；Stage 2 的五次训练各有独立 optimizer、scheduler、Trainer local step 和 RNG。

| 新实验 | 对应3maps ID | 训练路线 | 首次初始化来源 | 辅助损失 | 本阶段／两阶段累计更新步数 | 评测任务 |
|---|---|---|---|---|---|---|
| `exp37` | J8 | Stage 1，联合 J8 | 原始 UniLIP + π0.5 | 均关闭 | 19550／19550，已完成 | 定位、离散生成、连续生成 |
| `exp37_s2_gen` | S0_G0 | Stage 2，生成单任务 | 旧 `exp32_gen@19500` 模型 | 均关闭 | local6000／25500，已完成 | 离散生成、连续生成 |
| `exp37_s2_loc` | S0_L0 | Stage 2，定位单任务 | 旧 `exp32_loc@19500` 模型 | 均关闭 | local6000／25500，已完成 | 定位 |
| `exp37_s2` | S0_J8 | Stage 2，联合对照 | `exp37@19550` 最终模型 | 均关闭 | local6000／25550，已完成 | 定位、离散生成、连续生成 |
| `exp37_s2_both` | **B2_S3_J8** | Stage 2，联合辅助 | 同一 `exp37@19550` 最终模型 | P4 + U2，attention=false | local6000／25550，已完成 | 定位、离散生成、连续生成 |
| `exp37_s2_both_b1` | **B1_S3_J8** | Stage 2，联合辅助，独立 B1 候选 | 同一 `exp37@19550` 最终模型 | P3 + U2，attention=true | 计划local6000／25550；仅启动日志，完成状态待核实 | 定位、离散生成、连续生成 |

2026-11-07身份核对：三图 L0／G0 的 Seen-10 对应基座分别为复用的旧 `exp32_loc@19500`／`exp32_gen@19500`，不另建 `exp37` 单任务 Stage 1。三图 Stage 2 的 local1800及累计7650，对应本节的local6000及单任务累计25500／联合累计25550；累计步数仅表示训练预算，推理参数仍传本阶段checkpoint step。完整8组映射、辅助ramp 120→400、warmup 60→200和基础样本曝光口径见[三图简表身份核对](CSGO_BENCHMARK_V2_EXP32_3MAPS.md#experiment-id-map-3maps-seen10-20261107)。B1/B2都从联合Stage 1独立开始，P3/P4/U2是配方来源，`S3`不表示第三阶段。

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

### 2026-11-06：批量推理

`scripts/run_exp37_seen10.py infer/metrics` 新增以下参数；`metrics` 使用它们选择对应推理结果目录，并不改变指标计算器自身的 batch。旧的已报告结果保持原样。

如需分别补跑三项 test：`--task localization` 仅定位，`--task gen` 仅离散生成，`--task gen_continuous` 仅连续生成（只允许 test）。原 `--task generation` 在 test 仍执行两类生成，`--task all` 保持原任务分发。独立任务只读取／校验对应快照和结果，不要求其他任务已完成。`exp37` 最终 checkpoint 的六条独立命令见 [record.md](record.md#exp37-batched-inference-20261106)；定位 metrics 校验 infer 已生成的定位摘要，不重复计算。

| 参数 | 默认值 | 范围 |
|---|---:|---|
| `--test-loc-batch-size` | 32 | flow 定位同时推理的样本数 |
| `--test-gen-batch-size` | 16 | 离散与连续生成同时推理的样本数 |

生成对每批同时执行多模态编码、CFG、20 步 DiT 和 VAE 解码；定位沿用 10 步 flow 求解。连续生成仍是逐帧独立条件生成，不增加历史帧输入，不改变 clip 划分。最后不足一个 batch 的样本照常处理。两任务的批量值可分别调整，训练有效 batch、学习率、checkpoint 和评测定义均不受影响。

默认结果写入 `outputs_loc/benchmark_v2/{experiment}/checkpoint-{step}/{split}/seed_42/infer_bs32/` 与 `outputs_eval/benchmark_v2/{experiment}/checkpoint-{step}/{split}/seed_42/infer_bs16/{discrete,continuous}/`。每项保留独立 `evaluation_config.yaml`；执行清单记录实际 batch、样本顺序哈希、噪声策略、吞吐和 CUDA 峰值 allocated 显存。`infer` 与 `metrics` 必须传相同的批量值。显式设置两个参数为 `1` 时，使用原目录和单样本路径，可继续评测已有结果；省略参数则使用新的 32／16 默认值，**不会自动读取旧结果**。

批量模式按原数据顺序逐样本抽取初始噪声后拼成 batch，保留原设备、dtype、seed 和采样器。另有一个已核实的既存行为：生成入口在加载模型后注入 LoRA，当前 PEFT 下新增的 Dropout 可仍处于 training 模式；本次保留加载后的模块模式，未自动调用 `model.eval()` 改变旧行为。B>1 清单记录实际启用的 Dropout 模块，日志也会提示。**初始噪声顺序相同不等于完整输出相同**：Dropout mask 的随机数分配会随 batch 改变，此外还有浮点差异；应固定批量比较新结果，不将批量结果直接覆盖到旧单样本结果表。真实权重的误差与提速幅度尚未实测。是否统一关闭生成 Dropout，应另行作为明确的推理协议修正评估。

当前批量生成限定已核验的 Sana DPM++ flow scheduler；历史 token 定位头需显式使用 `--test-loc-batch-size 1`。显存不足时退出，不自动缩小 batch 或混合结果；手动减小批量值后会进入独立目录。中断后重跑同一 infer 会重算该任务，未实现逐样本跳过。

可执行模板见 [record.md 批量推理说明](record.md#exp37-batched-inference-20261106)。本次仅执行 CPU 回归；本机 GPU 已被现有任务占用，真实权重的 batch=1 对照、默认 32／16 显存及吞吐仍待空闲 GPU 验证。没有启动完整推理或训练。

验收：相关 55 项 CPU 回归通过，覆盖真实批处理方法的 CPU 替身执行、CFG／双图像／pose 对齐、padding 与尾批、初始噪声抽取、旧 B1 路径、配置及指标路由、缺失定位摘要拒绝、Dropout 状态记录无副作用；18 个改动 Python 文件语法检查、5 段新命令 `bash -n` 及 `git diff --check` 通过。扩展历史 runner 检查仍有 7 项因既有 checkpoint 缺失、资产配置冲突或结果文档格式不匹配未通过，不作为本次功能已验证的范围。

| 实验 | validation | 固定正式 test checkpoint | 当前结果 |
|---|---|---|---|
| `exp37` | 8000、实际 final=19550，已评 | Stage 1 `checkpoint-19550`，仅参照 | 训练完成；定位、离散及连续 test 已评 |
| `exp37_s2_gen` | local 4000、6000，已评 | local `checkpoint-6000` | 训练完成；原batch1 test已归档；本次batch8队列未执行，本机无结果 |
| `exp37_s2_loc` | local 4000、6000，已评 | local `checkpoint-6000` | 训练完成；原batch1 test已归档；本次batch8队列未执行，本机无结果 |
| `exp37_s2` | local 4000、6000，已评 | local `checkpoint-6000` | 训练完成；原batch1 test已归档；batch8定位已评，生成推理完成但指标未评测（前置校验失败） |
| `exp37_s2_both` | local4000定位已评（batch1）、离散中断；local6000未同步 | local `checkpoint-6000` | 训练完成；test定位／离散／连续✅，三项均batch8 |
| `exp37_s2_both_b1` | local 4000、6000 | local 6000 | 本地无训练或评测产物，状态待核实 |

### 已完成组的 Seen-10 实测结果（seed 42）

下表为固定正式 **test**，十图等权宏平均。XY 为定位 `XY_Dist`（越低越好）；PSNR、LPIPS、FID 来自离散生成，FVD 来自连续生成。`—` 表示该单任务路线不评对应任务；“未评测”表示推理产物存在但指标计算未执行；“本队列未执行”表示本次队列未生成对应batch8产物。每个已填数值取自所链 JSON 的 `metrics_macro_map`，不以 validation 替代 test。Stage 1 的batch1只作训练阶段参照；Stage 2 主表统一登记batch8结果，原batch1结果归档于紧随其后的历史表。

| 实验与 checkpoint | 推理 batch | XY ↓ | PSNR ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | 原始摘要 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exp37@19550` | 1 | 49.938 | 13.616 | 0.627 | 36.394 | 943.613 | [loc](outputs_loc/benchmark_v2/exp37/checkpoint-19550/test/seed_42/benchmark_csgo_v2_loc.json) · [discrete](outputs_eval/benchmark_v2/exp37/checkpoint-19550/test/seed_42/discrete/summary.json) · [continuous](outputs_eval/benchmark_v2/exp37/checkpoint-19550/test/seed_42/continuous/summary.json) |
| `exp37_s2_gen@6000` | 8 | — | 未评测 | 未评测 | 未评测 | 未评测 | 本队列未执行；无本机产物 |
| `exp37_s2_loc@6000` | 8 | 未评测 | — | — | — | — | 本队列未执行；无本机产物 |
| `exp37_s2@6000` | 8 | 47.173 | 未评测 | 未评测 | 未评测 | 未评测 | [loc JSON](outputs_loc/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json)；生成推理清单已核验，指标未执行 |
| `exp37_s2_both@6000` | 8 | 46.792 | 13.738 | 0.624 | 35.840 | 922.623 | [loc](outputs_loc/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json) · [discrete](outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/discrete/summary.json) · [continuous](outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/continuous/summary.json) |

历史记录：以下三组是2026-11-07已核对的原始batch1 test结果。保留原数值与来源，供追溯推理batch的影响；与上表batch8结果分开解释。

| 实验与 checkpoint | 推理 batch | XY ↓ | PSNR ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | 原始摘要 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exp37_s2_gen@6000` | 1 | — | 14.167 | 0.603 | 34.080 | 843.436 | [discrete](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/discrete/summary.json) · [continuous](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/continuous/summary.json) |
| `exp37_s2_loc@6000` | 1 | 46.353 | — | — | — | — | [loc](outputs_loc/benchmark_v2/exp37_s2_loc/checkpoint-6000/test/seed_42/benchmark_csgo_v2_loc.json) |
| `exp37_s2@6000` | 1 | 47.194 | 13.748 | 0.623 | 35.804 | 922.543 | [loc](outputs_loc/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/benchmark_csgo_v2_loc.json) · [discrete](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/discrete/summary.json) · [continuous](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/continuous/summary.json) |

validation 仅评定位和离散生成；连续生成按协议只在 test 评。以下数字同样为十图等权宏平均，`—` 为任务不适用：

| 实验与 validation checkpoint | XY ↓ | PSNR ↑ | LPIPS ↓ | FID ↓ |
|---|---:|---:|---:|---:|
| `exp37@8000` | 65.078 | 13.342 | 0.652 | 68.132 |
| `exp37@19550` | 41.955 | 13.892 | 0.617 | 63.872 |
| `exp37_s2_gen@4000 / 6000` | — / — | 14.487 / 14.578 | 0.589 / 0.585 | 62.286 / 61.203 |
| `exp37_s2_loc@4000 / 6000` | 39.560 / 38.709 | — / — | — / — | — / — |
| `exp37_s2@4000 / 6000` | 39.524 / 39.206 | 14.023 / 14.041 | 0.614 / 0.610 | 63.031 / 62.750 |
| `exp37_s2_both@4000 / 6000` | 41.328 / 未同步 | 未同步 / 未同步 | 未同步 / 未同步 | 未同步 / 未同步 |

validation 来源分别位于 `outputs_loc/benchmark_v2/<实验>/checkpoint-<步>/validation/seed_42/benchmark_csgo_v2_loc.json` 与 `outputs_eval/benchmark_v2/<实验>/checkpoint-<步>/validation/seed_42/discrete/summary.json`，其中 `<步>` 取表中实际 checkpoint。此前四组中，早期 validation 的 `exp37@8000` 和 Stage 2 `@4000` 评测 JSON 与 provenance 仍在，但对应模型权重未在本地保留；这四组正式终点 checkpoint 的权重和 Trainer 状态均在本地。新同步 `exp37_s2_both` 的 `@4000` 和最终 `@6000` 权重／状态也已在本地。训练完成性由 `outputs/csgo_1b/<实验>/checkpoint-<步>/trainer_state.json` 的 `global_step` 与同目录 `model.safetensors` 核对：Stage 1 为 19550，四组已完成 Stage 2 均为 local 6000。训练 `experiment_plan.json` 记录 `exp37_s2` 来源为 `exp37@19550`，两个单任务分别来自旧 `exp32_gen@19500` 和 `exp32_loc@19500`。

2026-11-07已核对的原batch1产物，其推理 provenance 指向 minimal 资产、seed 42、对应 checkpoint；validation 各任务 5,000 条，离散 test 各任务 20,000 条，连续 test 12,800 帧。历史batch1摘要覆盖完整十图，逐图生成指标与原始 `benchmark_csgo_v2_*.json` 一致，宏平均已复算。仅在这批历史batch1结果内部，`exp37_s2` 相对 Stage 1 参照的 test XY、PSNR、LPIPS、FID、FVD 均改善；其定位仍略逊于 `exp37_s2_loc`，生成各项仍逊于 `exp37_s2_gen`。Stage 1 与 Stage 2 训练预算不同，这里只描述观测差异，不归因为联合训练或辅助损失。截至2026-11-07，batch8对照仍在同步；之后队列状态与已评指标见下节。

结果写入 [`csgo_benchmark_v2_experiments_results.md`](csgo_benchmark_v2_experiments_results.md) 的 exp37 专区；Table 1 身份映射见 [`csgo_benchmark_v2_table1_seen10.md`](csgo_benchmark_v2_table1_seen10.md)。只有实际完成并核对 checkpoint、十图覆盖、后端、任务路由及宏平均后，才填指标和参数量。历史表格数值不因新增实验而改变。

<a id="exp37-bs8-queue-root-cause-20261108"></a>

## 2026-11-08：batch8队列停止原因与实际评测状态

日志 [`fst_outs/exp37_s2_controls.test_bs8.nohup.out`](fst_outs/exp37_s2_controls.test_bs8.nohup.out) 显示，`exp37_s2` 的定位、离散和连续推理均完成；随后队列在定位指标前置校验处报错：`Localization summary inference provenance differs`，并以 `set -e` 停止。定位 JSON 的指标有效：XY／Z／Pitch／Yaw为 **47.1732145309／2.8362459898／2.651352644／26.31470890045**，表中按四位小数展示。清单覆盖完整十图、样本数／checkpoint／seed／batch及payload均通过核对。摘要 provenance 保存的清单路径为 `/data/home/jiahao/UniLIP_outputs/outputs_loc/.../inference_manifest.json`；[`aggregate_csgo_benchmark_v2_metrics.py`](scripts/aggregate_csgo_benchmark_v2_metrics.py#L907) 写入前将路径 `expanduser().resolve()`，runner 在 [`run_exp37_seen10.py`](scripts/run_exp37_seen10.py) 比较时只对摘要值调用 `path_in_repo`、未对期望清单路径做同样的 `resolve()`。两者经文件系统核验指向同一文件（`samefile=True`），但规范化不对称导致校验误报；不是 `abspath` 自身解析符号链接，也不是定位指标无效或生成推理未完成。

`exp37_s2` 离散与连续生成均完成20,000／12,800条推理，清单中的实际文件名与十图manifest完全一致（缺失=0、多余=0）；但没有逐图指标文件或summary，指标子进程未执行，因此记“未评测”，不填造指标。队列在首个实验的指标前置校验退出，故本队列没有执行 `exp37_s2_loc`、`exp37_s2_gen` 的batch8推理或评测；仅据本日志与本机产物登记，不推断其他可能存在的远端进程。该段记录最初中断状态。后续已按用户要求修复exp37／exp38的路径身份比较：统一解析两侧符号链接，保留payload与协议校验，并补齐exp38缺失的摘要校验模块导入。既有结果JSON无需改写；修复不代表剩余评测已执行。恢复命令见[record.md](record.md#exp37-s2-controls-test-bs8-recovery-20261108)。

## 2026-11-08：batch8 对照已核验部分

同为local6000、推理batch8、minimal资产、seed42的Seen-10 test定位摘要已核验：无辅助联合`exp37_s2`的XY／Z／Pitch／Yaw依次为47.1732／2.8362／2.6514／26.3147；辅助联合`exp37_s2_both`依次为46.7915／2.7853／2.7759／26.0181。相对无辅助联合，both的XY、Z、Yaw误差分别下降0.81%、1.80%、1.13%，Pitch误差上升4.70%；定位呈指标间取舍，不构成全面提升。无辅助联合的生成指标尚待计算，两个单任务的batch8推理／评测尚待按恢复命令执行，暂不据此判断生成收益或与单任务的同batch排名。

<a id="exp37-s2-both-update-20261107"></a>

## 2026-11-07 历史记录：exp37_s2_both 同步进度、最终test与分析

本节记录当日仅有both的batch8、其他对照只有batch1时的分析。2026-11-08开始同步三组batch8对照后，本节“缺少同batch对照”的判断仅适用于当日状态；当前同batch比较以本页最新结果表及后续分析为准。

本组来源为 `exp37@19550`，配方为 P4＋U2（perception attention=false、αloc固定8；perception与aux-loc两项辅助权重在local0→400分别ramp至0.05和1）。已同步[训练计划](outputs/csgo_1b/exp37_s2_both/experiment_plan.json)、[训练配置](outputs/csgo_1b/exp37_s2_both/resolved_config.yaml)及[最终Trainer状态](outputs/csgo_1b/exp37_s2_both/checkpoint-6000/trainer_state.json)：global_step=max_steps=6000，实际双卡microbatch4／GAS16、有效基础样本batch128。根目录与checkpoint-6000的模型文件均存在，文件大小与safetensors头声明一致。训练完成与各项评测完成分开记录。

此前已完成 **local4000、seen_validation、定位**：完整十图共5000条，minimal资产、seed42、batch=1，指标内嵌provenance与[推理清单](outputs_loc/benchmark_v2/exp37_s2_both/checkpoint-4000/validation/seed_42/inference_manifest.json)一致；下表为逐图指标复算的等地图宏平均。

| 实验（validation@4000） | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 原始指标 |
|---|---:|---:|---:|---:|---|
| `exp37_s2_loc` | 39.5600 | 2.6484 | 2.3485 | 20.5830 | [JSON](outputs_loc/benchmark_v2/exp37_s2_loc/checkpoint-4000/validation/seed_42/benchmark_csgo_v2_loc.json) |
| `exp37_s2` | 39.5235 | 2.5778 | 2.5489 | 21.8859 | [JSON](outputs_loc/benchmark_v2/exp37_s2/checkpoint-4000/validation/seed_42/benchmark_csgo_v2_loc.json) |
| `exp37_s2_both` | 41.3281 | 2.7120 | 2.7387 | 21.3078 | [JSON](outputs_loc/benchmark_v2/exp37_s2_both/checkpoint-4000/validation/seed_42/benchmark_csgo_v2_loc.json) |

与同来源、同节点的无辅助 `exp37_s2@4000` 相比，both 的XY／Z／Pitch误差分别上升4.57%／5.21%／7.45%，Yaw误差下降2.64%。当前仅显示朝向中的Yaw有所改善，不能表述为定位全面提升；相较单任务 `exp37_s2_loc@4000`，四项误差均更高。这一结论仅描述相同batch1下的4000步validation；新到的6000步test另列如下，不跨split解释训练曲线。

[旧运行日志](fst_outs/exp37_s2_both.nohup.out)记录的4000步validation离散生成仍只有1290/5000张图片，并因SIGKILL中断，不能记为评测完成。local6000的validation定位／离散也尚未同步。这些缺项不影响单独登记已经完成的最终test。

### 本次新增：local6000 test，推理batch=8

三项均为minimal资产、seed42、完整Seen-10：定位20000条、离散20000张、连续12800帧（200个64帧clip）。三份推理清单均记录batch8和对应checkpoint；生成逐图coverage=1、缺失／多余预测=0。十图原始JSON与summary的逐图值、等地图宏平均均已核对。FID／FVD为逐图计算后的等权平均；连续FVD窗口16、stride16、size224沿用原协议。

**定位**

| 推理batch | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 来源 |
|---:|---:|---:|---:|---:|---|
| 8 | 46.7915 | 2.7853 | 2.7759 | 26.0181 | [JSON](outputs_loc/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json) |

**离散生成**

| 推理batch | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | Pixel MAE↓ | 来源 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 13.7378 | 0.4055 | 0.6238 | 0.5092 | 35.8402 | 0.8328 | 41.3240 | [JSON](outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/discrete/summary.json) |

**连续生成**

| 推理batch | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 来源 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 14.0884 | 0.4116 | 0.6082 | 0.5195 | 37.1505 | 43.2095 | 17.2722 | 922.6230 | [JSON](outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/continuous/summary.json) |

**对照分析与边界。** 当前 `exp37_s2`、`exp37_s2_loc`、`exp37_s2_gen` 的final test仅有batch1结果，未找到匹配batch8的完整对照。新增生成清单实际记录438个活跃Dropout模块及 `stochastic_layers_batch_dependent=true`；因此跨batch的差异不能只当作浮点误差，也不能反过来断言全部由batch造成。以下只描述已报告数值，不据此排名或证明辅助loss增益。

- 相较无辅助联合的旧batch1结果，both的XY／Z／Yaw误差分别低0.85%／1.88%／1.21%，Pitch高4.78%，表现为指标间取舍。相较定位单任务的旧结果，XY仍高0.95%，只有Z误差更低；不支持定位全面超过单任务。
- 生成数值与无辅助联合非常接近：离散PSNR差−0.0103 dB、FID差+0.0365，连续FVD为922.6230 vs 922.5427，差+0.0802。可表述为观测值约持平，不能认定严格等效或已经证明辅助loss无效。
- 与生成单任务的旧结果相比，离散PSNR为13.7378 vs 14.1668、FID为35.8402 vs 34.0804，连续FVD为922.6230 vs 843.4360；本次没有呈现超过单任务的证据，跨batch比较仍须保留上述限制。
- 下一步优先同步缺失的validation；若需判断辅助收益，应预先固定指标，在相同checkpoint、推理batch与模型模式下补必要的validation对照。B1尚无本机结果，最终辅助方案不能根据本次test重新选取。无需为本次结果登记重训或立即全面重跑test。

本次只整理已完成产物，未启动训练、推理或指标计算。

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
