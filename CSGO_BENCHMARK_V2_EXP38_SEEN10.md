# CSGO Benchmark v2：exp38 两阶段与 exp39 Stage 1 Seen-10 实验

## 设计方案（2026-11-07，先记录后实现；辅助配方按用户最新指令改为P4＋U2）

新增 `exp38_s2_joint`（联合无辅助）与 `exp38_s2_both`（联合＋P4＋U2）两组。两组均复用现有 `exp37@12000` 模型权重，再训练 Stage 2 local7550步，研究路线总更新次数为 **12000＋7550＝19550**。本次只实现代码、配置与手动命令，不启动训练、推理或评测。

来源已只读核对：`outputs/csgo_1b/exp37/checkpoint-12000/` 中存在 `model.safetensors`、`config.json`、`trainer_state.json`；状态为 `global_step=12000`、`max_steps=19550`、`epoch≈30.6905`。它是指定中间checkpoint，不能要求已训练到原Stage 1终点，也不能自动替换为 `exp37@19550`。

| 项目 | `exp38_s2_joint` | `exp38_s2_both` |
|---|---|---|
| Stage 1模型来源 | `exp37/checkpoint-12000/model.safetensors` | 相同文件，绑定同一SHA-256 |
| 训练任务 | balanced paired定位＋生成 | 相同 |
| Stage 2方法 | 联合无辅助对照 | 联合＋P4 perception＋U2 aux-loc |
| 主定位权重 | αloc恒定8 | 相同 |
| perception | 关闭 | `[0,503] → [0,0.05]`，之后恒定0.05；attention=false |
| aux-loc | 关闭 | `[0,503] → [0,1]`，之后恒定1；combined、2候选、candidate_tau=1 |
| Stage 2更新次数 | local7550 | local7550 |
| 总更新次数 | 12000＋7550＝19550 | 12000＋7550＝19550 |
| checkpoint／test节点 | local4000、7550（final） | 相同 |
| 对应累计节点 | 16000、19550 | 相同 |

两组从同一来源**仅加载模型权重**，分别新建AdamW、scheduler、Trainer local step与RNG。首次启动不是恢复 `exp37` 训练；`--resume` 仅恢复新实验自己的完整checkpoint。两组各用独立输出目录，源checkpoint保持原样。

### 优化设置与缩放

以现有 `exp37_s2`／`exp37_s2_both` 的6000步Stage 2为参照，只改变来源、Stage 2长度及相应时间尺度。缩放比为7550/6000；warmup与辅助ramp分别独立四舍五入：

- 学习率warmup：`round(200×7550/6000)=252` local steps。
- 辅助权重ramp：`round(400×7550/6000)=503` local steps；不将其改为2×252。
- 普通可训练参数／LoRA峰值LR=`2e-5`；活动定位头connector／norm／projector峰值LR=`1e-4`。
- AdamW：β=(0.9,0.999)、eps=1e-8、weight decay=0、clip norm=1；`cosine_with_min_lr`、`min_lr_rate=0.1`，终点LR分别为2e-6／1e-5。
- 有效基础样本batch=128；默认单卡microbatch4、GAS32，改变GPU数时相应调整GAS。两臂使用相同训练布局。LoRA、主任务结构、seed42及其他P4／U2细节沿用现有配方。
- P4：当前定位头teacher、无EMA、teacher detach，vision-tower token SmoothL1，attention=false、teacher_gt、`linear_1m_sigma`。
- U2：combined EM＋uncertainty、2候选、`candidate_tau=1`、`unc_tau=1`、`unc_min_weight=0.05`、共享定位噪声、`exp_sigma` λ=5、不额外renormalize。

新runner固定 `max_steps=7550`。在local4000暂停不会把scheduler改成4000步；恢复后继续原7550步调度，不重新warmup、不重新应用来源模型初始化。命令中的epoch上限不表示Stage 2额外训练50轮。

### 数据与评测

沿用完整Seen-10：训练50000条基础三元组；test定位／离散生成各20000条，连续生成200×64=12800帧，十图等权宏平均。默认minimal资产；首次训练可切换为full资产，后续resume／infer／metrics使用本实验保存的资产后端与路径。

两组、两个节点均做test定位、离散生成和连续生成。默认固定 `--test-loc-batch-size 32`、`--test-gen-batch-size 16`；metrics传入相同值以选中同一推理目录。支持分别执行 `localization`、`gen`、`gen_continuous`，`all`包含三项。定位指标由infer生成，metrics校验其摘要；生成metrics逐图计算并汇总。连续生成沿用逐帧条件生成，不增加历史帧输入。

local4000是预先指定的中间test报告点，local7550是固定正式终点；不根据4000的test结果调参、早停、选择checkpoint或拼接不同节点的最佳指标。本轮不新增validation节点。生成Dropout与批量行为沿用现有推理实现；新两臂可在相同batch下比较，与历史batch1／8结果对照时保留推理设置差异。

### 保存、暂停与恢复

完整模型及恢复状态**只保存 `checkpoint-4000/` 和 `checkpoint-7550/`**，每份包含模型、optimizer／scheduler、各rank RNG和Trainer状态。单独使用周期4000保存不能保证额外保存非周期终点7550，因此使用精确节点列表。新配置显式设 `training_checkpoint_steps: [4000,7550]`、`save_final_model_at_output_root: false`；精确节点callback负责保存，`save_total_limit=2`。保留根目录配置、训练计划、轻量Trainer状态及日志，但不额外导出一份根目录完整模型／adapter。

推荐手动串行流程：

1. 首次训练到local4000后保存并停止（全程horizon仍为7550）。
2. 对4000执行test推理及评测。
3. 本实验 `--resume --stop-after-step 0`，从4000的完整状态继续到7550。
4. 对final=7550执行test推理及评测。

直接一次训练到7550时也必须仅保存4000、7550，之后可分别补两节点评测。没有更早的恢复checkpoint；4000后的中断可从4000续训。首次不能加 `--resume`，也不能从另一个exp38实验的checkpoint恢复。runner仅接受这两个评测节点，拒绝其他step及validation，不删除其他实验产物。各节点的三任务结果、manifest及评测配置保留在对应实验／checkpoint／test／seed／batch目录。

### 实验定位与对照

- **辅助增量**：同一local节点的 `exp38_s2_both` 对照 `exp38_s2_joint`。
- **等累计更新次数的两阶段方案**：final两组对照 `exp37@19550`；变化包含提前切阶段、LR路径与优化器重置，不能全部归因于辅助损失。
- 旧单任务 `exp32_gen/loc@19500` 比本方案少50更新（约0.256%），只能称近似更新预算。联合与辅助目标每步计算量不同，**不宣称等FLOPs或等GPU时间**。
- 相比原联合两阶段路线19550＋6000＝25550，新路线减少6000累计更新；复用已有12000权重时，此次每臂新增训练7550步，两臂合计15100步。论文方法成本仍包含12000步前缀，不能把共享前缀计为零。

## 变更方案

1. 新增独立入口 `scripts/run_exp38_seen10.py` 和两份同名训练YAML，复用现有安全工具／评测接口；保留exp37及Unseen-4的既有来源、终点和任务约束。
2. 来源预检仅接受exp37@12000且原horizon=19550，核对已保存recipe／plan、十图协议、模型结构、LoRA header和SHA-256；两臂固定相同来源。源资产布局不限制新运行选择minimal或full，但数据协议必须相同。
3. 在 `unilip/train/csgo_staged_training.py` 与 `train_csgo.py` 增加可选的精确保存列表及关闭根模型重复导出逻辑；未设置新字段时沿用旧行为。
4. 新runner提供validate／train／infer／metrics、默认只打印计划；只有用户手动传 `--execute` 才写实验快照并执行。4000检查保存状态的 `global_step=4000,max_steps=7550`，final要求7550/7550。
5. `record.md` 分别提供两组首次到4000、test、恢复到final、final test的直接执行命令，以及minimal／full切换、Slurm中断恢复说明；不会自动启动后台任务。
6. CPU检查覆盖来源错误拒绝、7550 horizon、252／503调度、两节点保存与保留、根模型跳过、4000暂停恢复、三任务批量路由、minimal／full及只读dry run。既有exp37／Unseen入口与默认保存行为执行必要回归。

## 实施与实验进度

设计先于实现保存；代码、配置和手动命令已完成，both配方统一采用 **P4＋U2、perception attention=false**。截至2026-11-13复查，四个exp38分支仍仅同步local4000的test三任务结果，local7550最终结果尚未见同步；exp39与exp39_consloc12均完成19550步，16000／19550的test三任务结果完整，19550实测值已回填下方[final综合对照](#exp38-exp39-final-20261112)。exp37@16000的生成评测此前已补齐。下列CPU验收为此前实现阶段的记录；本次仅复核本机同步产物并更新文档，不据本机缺项判断远端是否完成。

- 新入口：[run_exp38_seen10.py](scripts/run_exp38_seen10.py)；配置：[joint](csgo_configs/exp38_s2_joint.yaml)、[both](csgo_configs/exp38_s2_both.yaml)。`validate` 和不带 `--execute` 的命令只读；首次train默认在4000停下，`--stop-after-step 0` 可一次跑完7550。
- 新runner的9项CPU测试通过；精确保存／暂停恢复的8项CPU测试通过，使用临时目录中的合成小模型确认同一horizon、两节点完整恢复材料、跨epoch无额外checkpoint及不重复导出根模型。既有exp37入口23项、Unseen-4入口13项回归通过，共53项。
- 16组训练计划＋64组infer／metrics计划核验通过，覆盖minimal／full、单卡／双卡、两个test节点、三项任务和默认batch32／16；验证未创建任何exp38的训练／推理／评测输出目录。两份nohup脚本通过 `bash -n`，保留record.md原有CRLF。
- 对真实来源进行了只读预检：12000/19550状态、2830个张量／1128个LoRA张量、模型文件4,577,302,678字节；SHA-256为 `0388328dc2e260bca15bb071fc4e9ff189fe631686fba553f187aea9917de8ae`。未将大模型载入GPU或执行项目训练／推理／评测；首次手动执行时runner仍会重新预检并将来源绑定到本实验计划。
- 新保存字段未配置时保留原训练保存行为；此条记录最初exp38实现时的兼容性验收。后续exp39路由及本次同步状态分别见下方设计章节和进度表。

| 实验 | 训练／本机checkpoint证据 | test累计16000 | test累计19550 | validation |
|---|---|---|---|---|
| `exp38_s2_joint` | 评测确认达到local4000；训练权重／Trainer状态未同步 | local4000三任务完成；batch32／16 | local7550未见同步结果；最终训练状态待核查 | 协议不新增 |
| `exp38_s2_both` | 本机4000/7550；4000模型、计划及状态已同步 | local4000三任务完成；**batch16／8** | local7550未见同步结果 | 协议不新增 |
| `exp38_s2_joint_constloc4` | 本机4000/7550；4000模型、计划及状态已同步 | local4000三任务完成；batch32／16 | local7550未见同步结果 | 协议不新增 |
| `exp38_s2_joint_constloc12` | 本机4000/7550；4000模型、计划及状态已同步 | local4000三任务完成；batch32／16 | local7550未见同步结果 | 协议不新增 |
| `exp39` | 19550/19550完成；12000／16000／19550模型和状态已同步 | 三任务完成；batch32／16 | 三任务完成；batch32／16 | 两节点未见同步结果 |
| `exp39_consloc12` | 19550/19550完成；12000／16000／19550模型和状态已同步 | 三任务完成；batch32／16 | 三任务完成；batch32／16 | 两节点未见同步结果 |
| `exp37`（Stage 1 α8对照） | 已完成19550；12000／16000／19550权重已有 | **三任务推理及评测均完成**；batch32／16 | 历史batch1结果已有；同batch32／16对照尚缺 | 16000未见结果；final历史结果已有 |

表中batch按“定位／生成”记载。joint@4000的状态依据推理manifest、评测摘要和evaluation_config；本机仍无 `outputs/csgo_1b/exp38_s2_joint/`，不能据此确认local7550是否训练完成。“未见”仅指本机同步证据。执行命令见 [record.md](record.md#exp38-seen10-20261107)。

## 中间节点对照：累计16000步 test（2026-11-12 更新）

按指定对照节点，已同步并核对 `exp32_loc@16000` 的 Seen-10 test 定位和 `exp32_gen@16000` 的 test 离散／连续生成结果，**三个任务均已完成**。推理manifest与结果摘要均绑定各自的 `checkpoint-16000/model.safetensors`；两份checkpoint的 `trainer_state.json` 均为 `global_step=16000,max_steps=19500`。定位／离散各20000条，连续12800帧，均覆盖完整十图。

历史19500步最终结果继续保留；下表区分其身份，本节16000对照表使用新同步的独立结果目录：

| 已有结果 | manifest／summary 绑定的权重 | 已有 test 范围 | 本次处理 |
|---|---|---|---|
| `outputs_loc/benchmark_v2/exp32_loc/seen/benchmark_csgo_v2_loc.json` | `outputs/csgo_1b/exp32_loc/model.safetensors`，根目录状态为19500 | `seen_discrete_test`，10图／20000条 | 保留原最终结果；16000定位已另行补齐 |
| `outputs_eval/benchmark_v2/exp32_gen/seen/discrete/summary.json` | `outputs/csgo_1b/exp32_gen/model.safetensors`，根目录状态为19500 | `seen_discrete_test`，10图／20000条 | 保留原最终结果；16000离散生成已另行补齐 |
| `outputs_eval/benchmark_v2/exp32_gen/seen/continuous/summary.json` | 同上 | `seen_continuous`，10图／12800帧 | 保留原最终结果；16000连续生成已另行补齐 |

**比较口径（2026-11-09 更正累计步数）**：exp38 的 Stage 2 local4000 对应累计 `12000+4000=16000` 次更新，与 exp32 单任务16000、exp37／exp39单阶段16000的更新次数相同。原文15000及多1000次更新的说明是算术笔误；不涉及checkpoint或训练配置变更。相同更新次数不等于相同FLOPs或GPU时间。辅助收益仍首先由同节点的 `exp38_s2_both` 与 `exp38_s2_joint` 判断，不根据中间test结果选择checkpoint。

下方exp32单任务对照使用seed42、定位batch32、生成batch16；新增结果仅exp38_s2_both为定位batch16／生成batch8，已在表中独立标明。资产后端为minimal，generation metrics使用 `benchmark_v2_core`、指标batch1、paired_size448、逐图评测后十图等权宏平均。连续生成沿用clip_length16／clip_stride16／fvd_size224。定位推理脚本同时输出定位评测，无需再执行一轮定位推理来计算指标。完整可执行命令及full资产切换说明见 [exp32_gen@16000](record.md#exp32-gen-test-16000) 和 [exp32_loc@16000](record.md#exp32-loc-test-16000)。

已核对十图逐图结果与摘要的宏平均值一致；生成逐图预测覆盖率均为100%，无缺失或额外预测。实测来源如下：

- 定位：[benchmark_csgo_v2_loc.json](outputs_loc/benchmark_v2/exp32_loc/checkpoint-16000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json)。
- 离散生成：[discrete/summary.json](outputs_eval/benchmark_v2/exp32_gen/checkpoint-16000/test/seed_42/infer_bs16/discrete/summary.json)。
- 连续生成：[continuous/summary.json](outputs_eval/benchmark_v2/exp32_gen/checkpoint-16000/test/seed_42/infer_bs16/continuous/summary.json)。

以下三表为累计16000次更新的完整实测对照，统一保留三位小数。新增推理batch列，**both行的batch与其余行不同，生成存在活跃Dropout，不能把跨batch差值直接归因于辅助loss**。定位／离散各20000条、连续12800帧；均为test、seed42、十图等权宏平均。训练／来源和最新分析见下文；不混入历史batch1／8的其他实验结果。

### test 定位对照

| 实验 | checkpoint | 累计训练step | 推理batch | 状态 | XY↓ | Z↓ | Pitch↓ | Yaw↓ |
|---|---|---|---|---|---|---|---|---|
| `exp32_loc` | 16000（单阶段） | 16000 | 32 | test推理／评测已完成 | 50.708 | 3.050 | 3.039 | 26.880 |
| `exp38_s2_joint` | Stage 2 local4000 | 16000 | 32 | test推理／评测已完成 | 52.309 | 2.974 | 3.236 | 27.945 |
| `exp38_s2_both` | Stage 2 local4000 | 16000 | 16 | test推理／评测已完成 | 54.076 | 3.248 | 2.954 | 27.791 |
| `exp38_s2_joint_constloc4` | Stage 2 local4000 | 16000 | 32 | test推理／评测已完成 | 51.578 | 3.030 | 2.890 | 27.858 |
| `exp38_s2_joint_constloc12` | Stage 2 local4000 | 16000 | 32 | test推理／评测已完成 | 50.624 | 3.016 | 2.667 | 28.138 |
| `exp39` | 16000（单阶段） | 16000 | 32 | test推理／评测已完成 | 49.459 | 3.072 | 2.883 | 26.914 |
| `exp39_consloc12` | 16000（单阶段） | 16000 | 32 | test推理／评测已完成 | 102.510 | 5.451 | 3.074 | 43.329 |
| `exp37` | 16000（单阶段） | 16000 | 32 | test推理／评测已完成 | 53.576 | 3.085 | 3.193 | 28.149 |

### test 离散生成对照

| 实验 | checkpoint | 累计训练step | 推理batch | 状态 | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ |
|---|---|---|---|---|---|---|---|---|---|
| `exp32_gen` | 16000（单阶段） | 16000 | 16 | test推理／评测已完成 | 13.815 | 0.410 | 0.616 | 0.514 | 36.167 |
| `exp38_s2_joint` | Stage 2 local4000 | 16000 | 16 | test推理／评测已完成 | 13.529 | 0.402 | 0.634 | 0.504 | 37.218 |
| `exp38_s2_both` | Stage 2 local4000 | 16000 | 8 | test推理／评测已完成 | 13.581 | 0.404 | 0.634 | 0.503 | 37.043 |
| `exp38_s2_joint_constloc4` | Stage 2 local4000 | 16000 | 16 | test推理／评测已完成 | 13.514 | 0.403 | 0.633 | 0.505 | 37.212 |
| `exp38_s2_joint_constloc12` | Stage 2 local4000 | 16000 | 16 | test推理／评测已完成 | 13.552 | 0.404 | 0.636 | 0.503 | 36.928 |
| `exp39` | 16000（单阶段） | 16000 | 16 | test推理／评测已完成 | 13.708 | 0.410 | 0.626 | 0.508 | 35.956 |
| `exp39_consloc12` | 16000（单阶段） | 16000 | 16 | test推理／评测已完成 | 13.799 | 0.414 | 0.621 | 0.510 | 35.719 |
| `exp37` | 16000（单阶段） | 16000 | 16 | test推理／评测已完成 | 13.605 | 0.409 | 0.634 | 0.503 | 36.456 |

### test 连续生成对照

| 实验 | checkpoint | 累计训练step | 推理batch | 状态 | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | TWE↓ | TDE↓ | OFE↓ | FVD↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `exp32_gen` | 16000（单阶段） | 16000 | 16 | test推理／评测已完成 | 14.098 | 0.412 | 0.605 | 0.523 | 36.496 | 42.628 | 17.365 | 867.691 |
| `exp38_s2_joint` | Stage 2 local4000 | 16000 | 16 | test推理／评测已完成 | 13.805 | 0.407 | 0.623 | 0.512 | 38.080 | 44.109 | 17.215 | 973.564 |
| `exp38_s2_both` | Stage 2 local4000 | 16000 | 8 | test推理／评测已完成 | 13.871 | 0.409 | 0.622 | 0.511 | 37.554 | 43.628 | 17.172 | 965.626 |
| `exp38_s2_joint_constloc4` | Stage 2 local4000 | 16000 | 16 | test推理／评测已完成 | 13.801 | 0.408 | 0.622 | 0.513 | 38.025 | 44.052 | 17.331 | 962.502 |
| `exp38_s2_joint_constloc12` | Stage 2 local4000 | 16000 | 16 | test推理／评测已完成 | 13.844 | 0.409 | 0.622 | 0.512 | 38.134 | 44.123 | 17.230 | 937.934 |
| `exp39` | 16000（单阶段） | 16000 | 16 | test推理／评测已完成 | 13.925 | 0.411 | 0.615 | 0.517 | 37.257 | 43.354 | 17.390 | 959.891 |
| `exp39_consloc12` | 16000（单阶段） | 16000 | 16 | test推理／评测已完成 | 14.106 | 0.416 | 0.606 | 0.521 | 36.298 | 42.453 | 17.269 | 915.466 |
| `exp37` | 16000（单阶段） | 16000 | 16 | test推理／评测已完成 | 13.910 | 0.415 | 0.620 | 0.512 | 37.339 | 43.445 | 17.204 | 969.281 |

### 2026-11-09：exp38_s2_joint@4000 同步记录与阶段性比较

本次同步为 **Stage 2 local4000，累计12000＋4000＝16000** 的联合无辅助对照。三个evaluation_config均记录来源 `exp37/checkpoint-12000/model.safetensors`、`exp38_experiment=exp38_s2_joint`、αloc=8、普通LR=2e-5、保存节点[4000,7550]；perception与aux-loc均关闭。推理manifest与摘要均绑定 `outputs/csgo_1b/exp38_s2_joint/checkpoint-4000/model.safetensors`。这些是评测产物中的身份信息，尚未通过本机训练checkpoint／Trainer状态做独立复核。

| 任务 | split／规模 | 实际推理设置 | 同步结果与核对状态 |
|---|---|---|---|
| 定位 | `seen_discrete_test`；10图／20000条 | seed42，batch32，10步flow推理 | [定位摘要](outputs_loc/benchmark_v2/exp38_s2_joint/checkpoint-4000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json)；完整，十图宏平均一致 |
| 离散生成 | `seen_discrete_test`；10图／20000条 | seed42，batch16，20步生成 | [离散summary](outputs_eval/benchmark_v2/exp38_s2_joint/checkpoint-4000/test/seed_42/infer_bs16/discrete/summary.json)；十图指标齐全，覆盖率100% |
| 连续生成 | `seen_continuous`；10图／12800帧（200×64） | seed42，batch16，20步生成 | [连续summary](outputs_eval/benchmark_v2/exp38_s2_joint/checkpoint-4000/test/seed_42/infer_bs16/continuous/summary.json)；十图指标齐全，覆盖率100% |

生成评测为 `benchmark_v2_core`、指标batch1、paired_size448，连续clip_length16／clip_stride16／fvd_size224；minimal资产后端与exp32@16000一致。逐图结果无缺失或额外预测，表中指标均已用十图等权宏平均复核。

相对上述同累计更新数、同推理batch和seed的exp32单任务对照，当前结果为：

- 定位仅Z改善（3.050→2.974）；XY增大1.601（50.708→52.309），Pitch增大0.197，Yaw增大1.065。
- 离散生成五项均未优于gen-only：PSNR下降0.286，SSIM下降0.007，LPIPS增大0.018，Boundary_F1下降0.010，FID增大1.051。
- 连续生成仅OFE改善（17.365→17.215）；PSNR下降0.293，LPIPS增大0.018，TWE／TDE分别增大1.584／1.481，FVD从867.691增至973.564（约+12.20%）。SSIM与Boundary_F1也下降。

因此，当前4000节点的联合无辅助路线在多数指标上仍未达到对应单任务表现。该比较同时包含联合目标与两阶段优化路径差异，不能只据此将退化归因于梯度冲突，也尚不能评价P4＋U2的辅助收益。后续按预定计划补齐同节点 `exp38_s2_both`、`exp37@16000`、constloc4对照及7550最终节点结果；不依据本次中间test调参或选择checkpoint。若远端尚未完成7550，按record中的本实验 `--resume --stop-after-step 0` 继续；若已经完成，只同步训练状态和缺失结果。

<a id="exp38-constloc4-results-20261110"></a>

### 2026-11-10：exp38_s2_joint_constloc4@4000 同步记录与阶段性比较

本次同步节点为 **Stage 2 local4000，累计12000＋4000＝16000**。根目录及checkpoint内的Trainer状态均为 `global_step=4000,max_steps=7550`；训练计划、resolved_config与4000模型文件已同步，配置指纹与计划一致。模型头部声明的长度与文件大小吻合，未载入模型或重新运行评测。**本机尚无7550 checkpoint及评测结果，不能把本次结果记为final。**

训练配方为exp37@12000权重初始化、主定位αloc=4、辅助损失关闭，普通LR=2e-5、action connector／norm／projector LR=1e-4、warmup252、基础batch128（1卡×micro64×GAS2）。来源在[训练计划](outputs/csgo_1b/exp38_s2_joint_constloc4/experiment_plan.json)中绑定SHA-256 `0388328dc2e260bca15bb071fc4e9ff189fe631686fba553f187aea9917de8ae`；训练状态见[Trainer状态](outputs/csgo_1b/exp38_s2_joint_constloc4/checkpoint-4000/trainer_state.json)。

| 任务 | 数据与实际推理设置 | 同步结果与核对状态 |
|---|---|---|
| 定位 | seen_discrete_test；十图各2000条；seed42、batch32、10步flow | [定位摘要](outputs_loc/benchmark_v2/exp38_s2_joint_constloc4/checkpoint-4000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json)；20000条，十图宏平均一致 |
| 离散生成 | seen_discrete_test；十图各2000张；seed42、batch16、20步生成 | [离散summary](outputs_eval/benchmark_v2/exp38_s2_joint_constloc4/checkpoint-4000/test/seed_42/infer_bs16/discrete/summary.json)；20000张，逐图指标与摘要一致，覆盖率100% |
| 连续生成 | seen_continuous；十图各20×64帧；seed42、batch16、20步生成 | [连续summary](outputs_eval/benchmark_v2/exp38_s2_joint_constloc4/checkpoint-4000/test/seed_42/infer_bs16/continuous/summary.json)；12800帧，逐图指标与摘要一致，覆盖率100% |

生成评测逐图GT／预测／匹配数量一致，无missing或额外预测；metric_profile=benchmark_v2_core、指标batch1、paired_size448，连续clip_length16／clip_stride16／fvd_size224。对比αloc=8的 `exp38_s2_joint@4000`，三任务的split、地图、样本数、seed、资产及选图SHA、样本顺序哈希、推理batch、步数与噪声策略一致；evaluation_config仅实验身份、目标checkpoint和αloc不同。两组生成沿用相同加载模式与批量推理实现，不混用历史batch1／8结果。

阶段性分析（相对差使用未舍入的宏平均计算）：

- **定位的主要收益是Pitch，XY也改善，但Z退步。** α8→4后，XY为52.309→51.578（−1.40%，7/10图改善），Pitch为3.236→2.890（−10.69%，10/10图改善），Yaw为27.945→27.858（−0.31%，6/10图改善）；Z为2.974→3.030（+1.90%，仅1/10图改善）。降低主定位权重并未使定位整体退化，也没有在四项误差上全面占优。
- **离散生成接近持平，没有明显恢复生成能力。** PSNR为13.529→13.514（−0.015），SSIM、LPIPS和Boundary_F1小幅改善；FID为37.218→37.212（−0.006）。这不足以支持“降低定位权重明显释放生成能力”的预期。
- **连续生成有部分小幅收益。** FVD为973.564→962.502（−1.14%，6/10图改善）；LPIPS、TWE、TDE略降，SSIM与Boundary_F1略升，但PSNR略降，OFE由17.215升至17.331（+0.67%）。不能概括为时序指标全面改善。
- **与同累计16000步的单任务相比，联合模型仍存在差距。** 定位相对exp32_loc：Pitch低4.90%、Z低0.64%，但XY高1.72%、Yaw高3.64%。离散生成五项主指标仍全部落后于exp32_gen；连续生成八项主指标中仅OFE略优，FVD为962.502，对照为867.691。当前α4主要改善了部分定位表现，尚未消除联合生成相对单任务的差距。

后续按既定计划完成或同步7550最终节点，并对照已经配置的α12和Stage 1 α4／12实验，观察阶段性变化是否延续。本次不据中间test结果新增调参、改变训练长度或指定最终Ours。仅更新文档，未启动训练、推理或评测。

<a id="exp38-exp39-results-20261111"></a>

## 2026-11-11：exp38／exp39 阶段性结果与分析（历史快照）

本节保留2026-11-11当时的同步状态与结论；其中缺项已在11-12补齐，当前状态以[最新记录](#exp38-exp39-results-20261112)为准。上方2026-11-09／10分析同样为历史记录。此次新增的是 **exp39@16000与@19550的完整test三任务，以及exp37@16000的test定位**。同日继续核查确认exp37@16000离散／连续生成推理产物完整，评测指标仍未同步，详见下方补充记录。exp38 joint α8／4仍只有local4000结果，both与两组α12实验未见本机训练或评测产物，远端状态不能由本机缺失推断。

### 训练身份与结果来源

exp39的根Trainer状态为 `global_step=max_steps=19550,epoch=50`，三个checkpoint状态分别为12000／16000／19550，均保持19550的scheduler horizon。保存的resolved_config与exp37仅差主定位权重8→4、实验身份及精确保存／关闭根模型副本字段；两者均从原始UniLIP＋π0.5初始化、关闭辅助loss，实际基础batch128（2卡×micro64×GAS1）、seed42、普通LR=1e-4、action connector／norm／projector LR=5e-4、warmup ratio=0.003及cosine设置一致。因此，**同step、同推理口径的exp39与exp37是Stage 1 αloc=4／8对照**，并非从exp37续训的Stage 2。exp39@12000仅保留为后续分叉来源，本次没有该节点评测结果。

| exp39节点 | 训练状态 | 定位摘要 | 离散生成摘要 | 连续生成摘要 |
|---|---|---|---|---|
| 16000 | [Trainer状态](outputs/csgo_1b/exp39/checkpoint-16000/trainer_state.json) | [loc JSON](outputs_loc/benchmark_v2/exp39/checkpoint-16000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [discrete JSON](outputs_eval/benchmark_v2/exp39/checkpoint-16000/test/seed_42/infer_bs16/discrete/summary.json) | [continuous JSON](outputs_eval/benchmark_v2/exp39/checkpoint-16000/test/seed_42/infer_bs16/continuous/summary.json) |
| 19550（final） | [Trainer状态](outputs/csgo_1b/exp39/checkpoint-19550/trainer_state.json) | [loc JSON](outputs_loc/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [discrete JSON](outputs_eval/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs16/discrete/summary.json) | [continuous JSON](outputs_eval/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs16/continuous/summary.json) |

新增exp37@16000对照来源：[定位JSON](outputs_loc/benchmark_v2/exp37/checkpoint-16000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json)。离散／连续目录现已同步生成图片及 `inference_manifest.json`，由“仅有执行计划”更新为“推理完成”；两目录均无逐图指标JSON和 `summary.json`，尚不能登记为生成评测完成。

两份exp39结果均为完整Seen-10：定位／离散各20000条，连续12800帧；seed42，定位batch32／10步flow，生成batch16／20步。宏平均已按十图等权复核，生成逐图GT／预测／匹配数一致、覆盖率100%，无missing或额外预测。资产清单SHA、选图SHA、样本顺序和噪声策略与本节16000对照一致；生成保留 `preserve_loaded_module_modes_v1`，活跃Dropout随batch改变，不能与历史batch1结果混作严格对照。生成指标沿用 `benchmark_v2_core`、指标batch1、paired_size448；连续clip_length16／clip_stride16／fvd_size224。exp39两个节点的validation尚未同步，本次全部数值均为test。

<a id="exp39-stage1-final-retained-20261112"></a>

### 已同步的exp39 Stage 1 final结果留档

以下保存此前同步并核对的Stage 1@19550结果，供原始记录和已有分析追溯。2026-11-13按最新要求重新核对源JSON后，已将两组结果回填后文final综合对照；本节保留相同实测值，训练与推理设置未变。

#### 留档：定位

| 实验／结果来源 | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|
| [exp39](outputs_loc/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | 47.050 | 2.863 | 2.713 | 26.119 | 32 | 19550 |
| [exp39_consloc12](outputs_loc/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | 95.145 | 5.157 | 3.132 | 41.284 | 32 | 19550 |

#### 留档：离散生成

| 实验／结果来源 | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|---|
| [exp39](outputs_eval/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs16/discrete/summary.json) | 13.839 | 0.414 | 0.621 | 0.511 | 35.400 | 16 | 19550 |
| [exp39_consloc12](outputs_eval/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs16/discrete/summary.json) | 13.926 | 0.411 | 0.614 | 0.516 | 35.530 | 16 | 19550 |

#### 留档：连续生成

| 实验／结果来源 | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | TWE↓ | TDE↓ | OFE↓ | FVD↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|---|---|---|---|
| [exp39](outputs_eval/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs16/continuous/summary.json) | 14.197 | 0.419 | 0.603 | 0.523 | 36.168 | 42.313 | 17.213 | 921.205 | 16 | 19550 |
| [exp39_consloc12](outputs_eval/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs16/continuous/summary.json) | 14.256 | 0.413 | 0.599 | 0.528 | 35.879 | 42.034 | 17.353 | 879.816 | 16 | 19550 |

### 分析结论

1. **从头使用αloc=4对定位的收益，比“α8前缀后再降为4”更值得关注。** 同16000步、batch32下，exp37→exp39的XY从53.576降至49.459（−7.69%，10/10图改善），Pitch从3.193降至2.883（−9.69%，9/10图改善），Yaw从28.149降至26.914（−4.39%，8/10图改善）；Z也略降0.43%，但仅4/10图改善。四项宏平均均较好，说明3maps选出的α8不应被视为Seen-10的既定最优值。当前可直接支持的是16000定位结论；exp37同节点生成推理已完成但指标尚缺，不能据此宣称α4已同时改善生成。
2. **Stage 1 α4优于当前“8→4”路线的多项指标，但这不是只改变前缀α的消融。** 同累计16000步，exp39对比exp38_s2_joint_constloc4：XY低4.11%、Yaw低3.39%，Pitch接近，Z高1.39%；离散五项全部改善，PSNR提高0.194、FID从37.212降至35.956；连续八项中七项改善，OFE略差，FVD仅降低0.27%。这支持继续核对从头调整主任务权重的方向，但两组还存在LR轨迹、优化器重置及阶段切换差异，不能把全部收益归因于“前12000步α4”。
3. **exp39从16000到固定final仍在改善，没有出现主指标整体回退。** 四项定位、五项离散、八项连续共17项宏平均全部改善：XY降低4.87%，Z降低6.83%，Pitch降低5.89%；离散PSNR提高0.131、FID降低1.55%；连续PSNR提高0.272、LPIPS降低1.98%、FVD降低4.03%（959.891→921.205）。定位XY／Pitch、离散PSNR／LPIPS、连续PSNR／LPIPS在十图均改善。这仅描述预先固定的两个报告点，不外推更长训练必然继续获益。
4. **联合模型与单任务的差距有所缩小，仍未实现全面不差于单任务。** 同16000步对exp32_loc，exp39的XY低2.46%、Pitch低5.12%，Z高0.74%、Yaw高0.13%。对exp32_gen，离散SSIM／FID略好，但PSNR低0.107、LPIPS高0.010、Boundary_F1低0.005；连续八项均较差，FVD高10.63%。FID改善不能替代配对图像与时序指标。本轮没有新同步的exp38 both结果，也不能对辅助loss是否解决Seen-10任务干扰作新结论。

### 后续优先级

1. **先补齐既定对照的评测／同步。** 优先补齐exp37@16000离散／连续的metrics（图片与推理manifest已齐，无需重跑infer）；远端已有评测时只同步逐图指标与summary。随后补齐exp37@19550定位batch32、生成batch16，以及exp38各分支local7550和两组α12的既定节点。缺失时再由用户执行record中的命令。对exp39@16000／19550与exp37对应节点补齐validation，供后续配方选择；不从当前test表选择各指标最优checkpoint。
2. **按两个问题分别比较。** Stage 1比较exp39(α4)、exp37(α8)、exp39_consloc12(α12)；固定α8前缀的Stage 2比较exp38 joint α4／8／12。当前Stage 1 α4值得优先保留为候选，但在同口径final和validation齐全前，不宣布它是最终Ours或放弃其余既定对照。
3. **后续辅助实验保持成对控制。** 若同口径证据继续支持α4基座，下一轮可考虑从同一exp39@12000出发的joint无辅助与P4＋U2两臂，保持来源、主权重、LR、长度、batch和推理设置一致，再评价辅助增量。它们应是独立新实验，不能直接替换现有exp38的来源；本次仅记录方向，未创建配置或启动实验。

本次复核新增38个表格指标单元格及40份逐图生成报告；exp39三个模型文件的头部声明长度与文件大小一致，未重算整份权重哈希或载入模型。只更新本结果文档，未修改训练／推理代码、配置或record中的用户命令，未运行模型和指标计算。

<a id="exp37-test-16000-sync-20261111"></a>

### 2026-11-11补充：exp37@16000 三任务推理与评测状态（历史快照）

2026-11-12更新：下表记录11-11当时的同步状态；离散／连续评测现已完成，当前指标见本文件累计16000步三任务表及[最新来源](#exp38-exp39-results-20261112)。

本次同步已补齐生成推理产物，但**生成评测指标仍缺失**。本机两个生成目录均未发现 `benchmark_csgo_v2_*.json` 或 `summary.json`；这表示本机尚无可登记的分数，不能推断远端是否已经执行metrics。定位指标已按十图等权宏平均重新核对，保持原值。

| 任务 | 推理口径 | 已同步样本 | 推理状态 | 评测状态与证据 |
|---|---|---|---|---|
| 定位 | seed42、batch32、10步flow | 20000条／10图 | 完成 | 完成；[定位JSON](outputs_loc/benchmark_v2/exp37/checkpoint-16000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json)：XY **53.576**、Z **3.085**、Pitch **3.193**、Yaw **28.149** |
| 离散生成 | seed42、batch16、20步 | 20000张／10图，每图2000 | 完成 | 指标未同步；[完成manifest](outputs_eval/benchmark_v2/exp37/checkpoint-16000/test/seed_42/infer_bs16/discrete/inference_manifest.json) |
| 连续生成 | seed42、batch16、20步 | 12800帧／10图，每图20×64 | 完成 | 指标未同步；[完成manifest](outputs_eval/benchmark_v2/exp37/checkpoint-16000/test/seed_42/infer_bs16/continuous/inference_manifest.json) |

两个生成目录的 `gen_imgs/<map>/` 文件名集合已逐图与正式discrete_test／continuous_clips划分核对：缺失0、额外0；这是文件覆盖核查，不是PSNR／FID／FVD等指标计算。推理manifest均指向 `exp37/checkpoint-16000/model.safetensors`，记录完成1250／800个batch；同步的 [exp37日志](fst_outs/exp37.nohup.out)末段也显示连续12800帧推理完成。batch、seed、样本顺序、资产指纹、推理步数、噪声策略及生成加载模式与exp39@16000、两个exp38 joint@4000一致，后续metrics补齐后可按既定口径对比。

本次没有新增生成分数，因此上方Stage 1 α4／8定位分析保持不变，暂不扩展为生成结论。未运行训练、推理或评测，未修改record命令。

<a id="exp38-exp39-results-20261112"></a>

## 2026-11-12：exp38／exp39／exp37@16000 同步结果与分析

本次补齐 **exp37@16000离散／连续指标、exp38_s2_both@4000三任务、exp38_s2_joint_constloc12@4000三任务，以及exp39_consloc12@16000／19550三任务**。旧实验数值继续保留，当前进度和累计16000步三表已更新；前文11-09～11-11的分析仅作历史快照。

### 实测来源与口径

| 实验 | 本地step | 累计step | 定位／生成batch | 定位摘要 | 离散摘要 | 连续摘要 |
|---|---|---|---|---|---|---|
| `exp37` | 16000 | 16000 | 32／16 | [loc](outputs_loc/benchmark_v2/exp37/checkpoint-16000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [discrete](outputs_eval/benchmark_v2/exp37/checkpoint-16000/test/seed_42/infer_bs16/discrete/summary.json) | [continuous](outputs_eval/benchmark_v2/exp37/checkpoint-16000/test/seed_42/infer_bs16/continuous/summary.json) |
| `exp38_s2_both` | 4000 | 16000 | 16／8 | [loc](outputs_loc/benchmark_v2/exp38_s2_both/checkpoint-4000/test/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) | [discrete](outputs_eval/benchmark_v2/exp38_s2_both/checkpoint-4000/test/seed_42/infer_bs8/discrete/summary.json) | [continuous](outputs_eval/benchmark_v2/exp38_s2_both/checkpoint-4000/test/seed_42/infer_bs8/continuous/summary.json) |
| `exp38_s2_joint_constloc12` | 4000 | 16000 | 32／16 | [loc](outputs_loc/benchmark_v2/exp38_s2_joint_constloc12/checkpoint-4000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [discrete](outputs_eval/benchmark_v2/exp38_s2_joint_constloc12/checkpoint-4000/test/seed_42/infer_bs16/discrete/summary.json) | [continuous](outputs_eval/benchmark_v2/exp38_s2_joint_constloc12/checkpoint-4000/test/seed_42/infer_bs16/continuous/summary.json) |
| `exp39_consloc12` | 16000 | 16000 | 32／16 | [loc](outputs_loc/benchmark_v2/exp39_consloc12/checkpoint-16000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [discrete](outputs_eval/benchmark_v2/exp39_consloc12/checkpoint-16000/test/seed_42/infer_bs16/discrete/summary.json) | [continuous](outputs_eval/benchmark_v2/exp39_consloc12/checkpoint-16000/test/seed_42/infer_bs16/continuous/summary.json) |
| `exp39_consloc12` | 19550 | 19550 | 32／16 | [loc](outputs_loc/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [discrete](outputs_eval/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs16/discrete/summary.json) | [continuous](outputs_eval/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs16/continuous/summary.json) |

所有新增摘要均覆盖完整Seen-10，定位／离散20000条，连续12800帧。宏平均由十图逐图结果复核；生成逐图GT／预测／匹配数均一致，缺失与额外预测均为0。定位10步flow、生成20步，seed42；资产指纹、样本顺序、噪声策略与既有对照一致。生成指标为CUDA上的benchmark_v2_core、指标batch1、paired_size448，连续clip_length16／clip_stride16／fvd_size224。

**实际推理batch例外为both：定位16、生成8；其余本轮对照定位32、生成16。** 生成manifest仍记录preserve_loaded_module_modes_v1和batch相关的活跃Dropout，故即使seed相同也不能把both与joint的差值视为严格的辅助loss消融。当前先如实记录，补齐同batch对照后再判断辅助增量；不覆盖已有结果。

训练证据另行核对：

- exp38 both／constloc12的本机状态均为4000/7550，4000模型及状态完整；两者与constloc4的计划均绑定同一exp37@12000来源，SHA-256为`0388328dc2e260bca15bb071fc4e9ff189fe631686fba553f187aea9917de8ae`。普通LR2e-5、warmup252、horizon7550。both主权重8，perception／aux-loc在[0,503]分别ramp至0.05／1，attention=false；配置初始辅助权重0不表示4000步时辅助关闭。constloc12无辅助、主权重12。
- 实际训练均维持基础batch128：both为2卡×micro4×GAS16，constloc12为2×64×1，constloc4为1×64×2。设计中“相同布局”是计划，实测布局以上述记录为准；本次不更改训练设置。
- exp39_consloc12完成19550/19550、epoch50，12000／16000／19550模型及状态齐全；与exp39的保存配方仅实验名和αloc=4→12不同，均从原始UniLIP＋π0.5初始化，无辅助，普通LR1e-4、warmup ratio0.003、基础batch128（2×64×1），没有Stage 2来源。定位退化不能解释为误将它记成exp38续训。
- exp38四分支仍未见local7550结果；exp39系列两个报告节点的validation仍未同步。exp37 final目前仍是历史batch1记录，同batch32／16终点对照尚缺。

<a id="exp38-exp39-final-20261112"></a>

### 最终节点：final test 综合对照（2026-11-13 更新）

本节按final汇总。按用户确认，**19500与19550之间由batch／梯度累计形成的50步差异视为允许误差**，不据此排除单任务对照；末列仍如实记录累计更新数。两阶段的25500／25550同样保留实际值，其6000步续训成本不会并入单阶段19500／19550预算。

每个任务表先列exp32、exp32_1、exp31三组历史联合配方，用分割线与后续实验隔开；单任务只列其适用任务。**推理batch固定为倒数第二列，累计训练steps固定为最后一列。** 实验名链接到实测JSON，配置列链接到对应YAML；所有实测值为完整Seen-10 test十图等权宏平均。

- exp37_s2_gen／loc：19500＋6000＝**25500**；exp37_s2／both／both_b1：19550＋6000＝**25550**。这里的6000只用于既有exp37_s2系列。
- 旧exp31／exp32系列与exp37采用历史batch1结果；“1（历史配置）”表示旧配置与记录的batch值，旧manifest未独立记录实际执行batch。exp37_s2／gen／loc／both使用batch8，both_b1定位batch32、生成batch16。推理batch与指标计算batch不同；各batch结果不混称严格同设置消融。exp37_s2_gen离散指标在CPU计算，其余所列Stage 2生成结果沿用各自已记录评测设置。
- **2026-11-13复查：exp39与exp39_consloc12的19550步test三任务均完整，已回填6行、34个指标值。** 摘要绑定各自`checkpoint-19550/model.safetensors`，seed42，实际定位batch32、生成batch16；定位／离散各20000条、连续12800帧，完整覆盖十图。已复核十图等权宏平均，生成覆盖率100%；数值与此前[原始记录](#exp39-stage1-final-retained-20261112)一致，这是已有结果的回填，并非新运行。
- **exp38四个分支的final指标继续留“—”**：本机仅见local4000结果，未见local7550（累计19550）或local7500（累计19500）的最终推理／评测结果；exp39也未见另一个19500步结果集。exp38占位行的batch仍是runner默认定位32／生成16，累计steps是既定预算，不代表已完成。4000步结果继续只列在累计16000步表。本次不修改任何训练计划，不新增6000步训练，也不将未同步解释为远端未完成。
- exp32_1的19550步沿用[此前训练状态核对](csgo_benchmark_v2_table1_seen10.md#exp37-定位最终test)；当前本机root训练状态缺失，三任务指标仍完整。exp32_gen／loc的19500步另由exp37_s2_gen／loc来源计划绑定确认。

#### test 定位

| 实验／结果来源 | 配置／训练方式 | checkpoint | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|---|---|
| [exp32](outputs_loc/benchmark_v2/exp32/seen/benchmark_csgo_v2_loc.json) | [LoRA联合，无辅助](csgo_configs/exp32.yaml) | final@19550 | 123.194 | 6.384 | 3.352 | 55.449 | 1（历史配置） | 19550 |
| [exp32_1](outputs_loc/benchmark_v2/exp32_1/seen/benchmark_csgo_v2_loc.json) | [LoRA联合，perception＋aux-loc](csgo_configs/exp32_1.yaml) | final@19550 | 63.038 | 3.560 | 3.088 | 31.671 | 1（历史配置） | 19550 |
| [exp31](outputs_loc/benchmark_v2/exp31/seen/benchmark_csgo_v2_loc.json) | [full-head联合，perception＋aux-loc](csgo_configs/exp31.yaml) | final@19550 | 99.097 | 4.960 | 3.219 | 45.110 | 1（历史配置） | 19550 |

---

| 实验／结果来源 | 配置／训练方式 | checkpoint | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|---|---|
| [exp37](outputs_loc/benchmark_v2/exp37/checkpoint-19550/test/seed_42/benchmark_csgo_v2_loc.json) | [Stage 1联合，αloc=8，无辅助](csgo_configs/exp37.yaml) | final@19550 | 49.938 | 2.998 | 2.675 | 26.865 | 1（历史配置） | 19550 |
| [exp32_loc](outputs_loc/benchmark_v2/exp32_loc/seen/benchmark_csgo_v2_loc.json) | [LoRA定位单任务](csgo_configs/exp32_loc.yaml) | final@19500 | 48.961 | 3.006 | 2.970 | 26.047 | 1（历史配置） | 19500 |
| [exp37_s2](outputs_loc/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json) | [exp37 final→联合续训，αloc=8，无辅助](csgo_configs/exp37_s2.yaml) | local6000（final） | 47.173 | 2.836 | 2.651 | 26.315 | 8 | 25550 |
| [exp37_s2_loc](outputs_loc/benchmark_v2/exp37_s2_loc/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json) | [exp32_loc final→定位单任务续训](csgo_configs/exp37_s2_loc.yaml) | local6000（final） | 46.341 | 2.875 | 2.517 | 25.153 | 8 | 25500 |
| [exp37_s2_both](outputs_loc/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/benchmark_csgo_v2_loc.json) | [exp37 final→联合P4＋U2；αloc=8](csgo_configs/exp37_s2_both.yaml) | local6000（final） | 46.792 | 2.785 | 2.776 | 26.018 | 8 | 25550 |
| [exp37_s2_both_b1](outputs_loc/benchmark_v2/exp37_s2_both_b1/checkpoint-6000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [exp37 final→联合P3＋U2；αloc=8](csgo_configs/exp37_s2_both_b1.yaml) | local6000（final） | 47.564 | 2.830 | 2.670 | 26.066 | 32 | 25550 |
| `exp38_s2_joint` | [exp37@12000→联合续训；αloc=8，无辅助](csgo_configs/exp38_s2_joint.yaml) | final@local7550（预留） | — | — | — | — | 32（默认配置） | 19550（既定预算） |
| `exp38_s2_both` | [exp37@12000→联合P4＋U2；αloc=8](csgo_configs/exp38_s2_both.yaml) | final@local7550（预留） | — | — | — | — | 32（默认配置） | 19550（既定预算） |
| `exp38_s2_joint_constloc4` | [exp37@12000→联合续训；αloc=4，无辅助](csgo_configs/exp38_s2_joint_constloc4.yaml) | final@local7550（预留） | — | — | — | — | 32（默认配置） | 19550（既定预算） |
| `exp38_s2_joint_constloc12` | [exp37@12000→联合续训；αloc=12，无辅助](csgo_configs/exp38_s2_joint_constloc12.yaml) | final@local7550（预留） | — | — | — | — | 32（默认配置） | 19550（既定预算） |
| [exp39](outputs_loc/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [原始初始化→Stage 1联合；αloc=4，无辅助](csgo_configs/exp39.yaml) | final@19550 | 47.050 | 2.863 | 2.713 | 26.119 | 32 | 19550 |
| [exp39_consloc12](outputs_loc/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json) | [原始初始化→Stage 1联合；αloc=12，无辅助](csgo_configs/exp39_consloc12.yaml) | final@19550 | 95.145 | 5.157 | 3.132 | 41.284 | 32 | 19550 |

#### test 离散生成

| 实验／结果来源 | 配置／训练方式 | checkpoint | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|---|---|---|
| [exp32](outputs_eval/benchmark_v2/exp32/seen/discrete/summary.json) | [LoRA联合，无辅助](csgo_configs/exp32.yaml) | final@19550 | 14.153 | 0.414 | 0.603 | 0.523 | 36.601 | 1（历史配置） | 19550 |
| [exp32_1](outputs_eval/benchmark_v2/exp32_1/seen/discrete/summary.json) | [LoRA联合，perception＋aux-loc](csgo_configs/exp32_1.yaml) | final@19550 | 13.907 | 0.407 | 0.615 | 0.515 | 36.216 | 1（历史配置） | 19550 |
| [exp31](outputs_eval/benchmark_v2/exp31/seen/discrete/summary.json) | [full-head联合，perception＋aux-loc](csgo_configs/exp31.yaml) | final@19550 | 14.627 | 0.426 | 0.585 | 0.529 | 28.304 | 1（历史配置） | 19550 |

---

| 实验／结果来源 | 配置／训练方式 | checkpoint | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|---|---|---|
| [exp37](outputs_eval/benchmark_v2/exp37/checkpoint-19550/test/seed_42/discrete/summary.json) | [Stage 1联合，αloc=8，无辅助](csgo_configs/exp37.yaml) | final@19550 | 13.616 | 0.404 | 0.627 | 0.508 | 36.394 | 1（历史配置） | 19550 |
| [exp32_gen](outputs_eval/benchmark_v2/exp32_gen/seen/discrete/summary.json) | [LoRA生成单任务](csgo_configs/exp32_gen.yaml) | final@19500 | 13.965 | 0.411 | 0.611 | 0.516 | 35.584 | 1（历史配置） | 19500 |
| [exp37_s2](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/infer_bs8/discrete/summary.json) | [exp37 final→联合续训，αloc=8，无辅助](csgo_configs/exp37_s2.yaml) | local6000（final） | 13.748 | 0.409 | 0.623 | 0.510 | 35.862 | 8 | 25550 |
| [exp37_s2_gen](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/infer_bs8/discrete/summary.json) | [exp32_gen final→生成单任务续训](csgo_configs/exp37_s2_gen.yaml) | local6000（final） | 14.169 | 0.416 | 0.603 | 0.522 | 34.324 | 8 | 25500 |
| [exp37_s2_both](outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/discrete/summary.json) | [exp37 final→联合P4＋U2；αloc=8](csgo_configs/exp37_s2_both.yaml) | local6000（final） | 13.738 | 0.405 | 0.624 | 0.509 | 35.840 | 8 | 25550 |
| [exp37_s2_both_b1](outputs_eval/benchmark_v2/exp37_s2_both_b1/checkpoint-6000/test/seed_42/infer_bs16/discrete/summary.json) | [exp37 final→联合P3＋U2；αloc=8](csgo_configs/exp37_s2_both_b1.yaml) | local6000（final） | 13.761 | 0.408 | 0.623 | 0.510 | 35.628 | 16 | 25550 |
| `exp38_s2_joint` | [exp37@12000→联合续训；αloc=8，无辅助](csgo_configs/exp38_s2_joint.yaml) | final@local7550（预留） | — | — | — | — | — | 16（默认配置） | 19550（既定预算） |
| `exp38_s2_both` | [exp37@12000→联合P4＋U2；αloc=8](csgo_configs/exp38_s2_both.yaml) | final@local7550（预留） | — | — | — | — | — | 16（默认配置） | 19550（既定预算） |
| `exp38_s2_joint_constloc4` | [exp37@12000→联合续训；αloc=4，无辅助](csgo_configs/exp38_s2_joint_constloc4.yaml) | final@local7550（预留） | — | — | — | — | — | 16（默认配置） | 19550（既定预算） |
| `exp38_s2_joint_constloc12` | [exp37@12000→联合续训；αloc=12，无辅助](csgo_configs/exp38_s2_joint_constloc12.yaml) | final@local7550（预留） | — | — | — | — | — | 16（默认配置） | 19550（既定预算） |
| [exp39](outputs_eval/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs16/discrete/summary.json) | [原始初始化→Stage 1联合；αloc=4，无辅助](csgo_configs/exp39.yaml) | final@19550 | 13.839 | 0.414 | 0.621 | 0.511 | 35.400 | 16 | 19550 |
| [exp39_consloc12](outputs_eval/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs16/discrete/summary.json) | [原始初始化→Stage 1联合；αloc=12，无辅助](csgo_configs/exp39_consloc12.yaml) | final@19550 | 13.926 | 0.411 | 0.614 | 0.516 | 35.530 | 16 | 19550 |

#### test 连续生成

| 实验／结果来源 | 配置／训练方式 | checkpoint | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | TWE↓ | TDE↓ | OFE↓ | FVD↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [exp32](outputs_eval/benchmark_v2/exp32/seen/continuous/summary.json) | [LoRA联合，无辅助](csgo_configs/exp32.yaml) | final@19550 | 14.572 | 0.419 | 0.583 | 0.538 | 34.092 | 40.360 | 17.405 | 848.459 | 1（历史配置） | 19550 |
| [exp32_1](outputs_eval/benchmark_v2/exp32_1/seen/continuous/summary.json) | [LoRA联合，perception＋aux-loc](csgo_configs/exp32_1.yaml) | final@19550 | 14.228 | 0.410 | 0.598 | 0.526 | 36.178 | 42.285 | 17.392 | 891.603 | 1（历史配置） | 19550 |
| [exp31](outputs_eval/benchmark_v2/exp31/seen/continuous/summary.json) | [full-head联合，perception＋aux-loc](csgo_configs/exp31.yaml) | final@19550 | 15.190 | 0.433 | 0.559 | 0.551 | 31.037 | 37.657 | 16.904 | 727.507 | 1（历史配置） | 19550 |

---

| 实验／结果来源 | 配置／训练方式 | checkpoint | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | TWE↓ | TDE↓ | OFE↓ | FVD↓ | 推理 batch | 累计训练 steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [exp37](outputs_eval/benchmark_v2/exp37/checkpoint-19550/test/seed_42/continuous/summary.json) | [Stage 1联合，αloc=8，无辅助](csgo_configs/exp37.yaml) | final@19550 | 13.882 | 0.407 | 0.617 | 0.515 | 37.958 | 43.978 | 17.413 | 943.613 | 1（历史配置） | 19550 |
| [exp32_gen](outputs_eval/benchmark_v2/exp32_gen/seen/continuous/summary.json) | [LoRA生成单任务](csgo_configs/exp32_gen.yaml) | final@19500 | 14.342 | 0.416 | 0.593 | 0.529 | 35.698 | 41.867 | 17.376 | 848.071 | 1（历史配置） | 19500 |
| [exp37_s2](outputs_eval/benchmark_v2/exp37_s2/checkpoint-6000/test/seed_42/infer_bs8/continuous/summary.json) | [exp37 final→联合续训，αloc=8，无辅助](csgo_configs/exp37_s2.yaml) | local6000（final） | 14.077 | 0.414 | 0.609 | 0.520 | 37.098 | 43.180 | 17.316 | 928.648 | 8 | 25550 |
| [exp37_s2_gen](outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/infer_bs8/continuous/summary.json) | [exp32_gen final→生成单任务续训](csgo_configs/exp37_s2_gen.yaml) | local6000（final） | 14.633 | 0.422 | 0.581 | 0.537 | 34.458 | 40.663 | 17.262 | 822.676 | 8 | 25500 |
| [exp37_s2_both](outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/continuous/summary.json) | [exp37 final→联合P4＋U2；αloc=8](csgo_configs/exp37_s2_both.yaml) | local6000（final） | 14.088 | 0.412 | 0.608 | 0.519 | 37.151 | 43.210 | 17.272 | 922.623 | 8 | 25550 |
| [exp37_s2_both_b1](outputs_eval/benchmark_v2/exp37_s2_both_b1/checkpoint-6000/test/seed_42/infer_bs16/continuous/summary.json) | [exp37 final→联合P3＋U2；αloc=8](csgo_configs/exp37_s2_both_b1.yaml) | local6000（final） | 14.101 | 0.413 | 0.608 | 0.521 | 37.168 | 43.202 | 17.299 | 906.544 | 16 | 25550 |
| `exp38_s2_joint` | [exp37@12000→联合续训；αloc=8，无辅助](csgo_configs/exp38_s2_joint.yaml) | final@local7550（预留） | — | — | — | — | — | — | — | — | 16（默认配置） | 19550（既定预算） |
| `exp38_s2_both` | [exp37@12000→联合P4＋U2；αloc=8](csgo_configs/exp38_s2_both.yaml) | final@local7550（预留） | — | — | — | — | — | — | — | — | 16（默认配置） | 19550（既定预算） |
| `exp38_s2_joint_constloc4` | [exp37@12000→联合续训；αloc=4，无辅助](csgo_configs/exp38_s2_joint_constloc4.yaml) | final@local7550（预留） | — | — | — | — | — | — | — | — | 16（默认配置） | 19550（既定预算） |
| `exp38_s2_joint_constloc12` | [exp37@12000→联合续训；αloc=12，无辅助](csgo_configs/exp38_s2_joint_constloc12.yaml) | final@local7550（预留） | — | — | — | — | — | — | — | — | 16（默认配置） | 19550（既定预算） |
| [exp39](outputs_eval/benchmark_v2/exp39/checkpoint-19550/test/seed_42/infer_bs16/continuous/summary.json) | [原始初始化→Stage 1联合；αloc=4，无辅助](csgo_configs/exp39.yaml) | final@19550 | 14.197 | 0.419 | 0.603 | 0.523 | 36.168 | 42.313 | 17.213 | 921.205 | 16 | 19550 |
| [exp39_consloc12](outputs_eval/benchmark_v2/exp39_consloc12/checkpoint-19550/test/seed_42/infer_bs16/continuous/summary.json) | [原始初始化→Stage 1联合；αloc=12，无辅助](csgo_configs/exp39_consloc12.yaml) | final@19550 | 14.256 | 0.413 | 0.599 | 0.528 | 35.879 | 42.034 | 17.353 | 879.816 | 16 | 19550 |

### 更新后的分析

1. **Stage 1 α4相对α8的生成对照现已补齐，收益主要成立但并非所有指标占优。** 同16000步，exp39相对exp37的四项定位误差均更低，XY为53.576→49.459（−7.69%），Pitch为3.193→2.883（−9.69%）；离散五项均改善，PSNR为13.605→13.708、FID为36.456→35.956。连续八项中六项改善，FVD为969.281→959.891（−0.97%），但SSIM由0.415降至0.411、OFE由17.204升至17.390。因此可以更新为“α4改善了定位和大部分生成指标”，不能写成三任务全面最优。
2. **Stage 1 α12显著损害定位，生成改善不足以支持统一模型采用该配方。** 同16000步相对α4，α12离散五项、连续八项宏指标均更好，但XY为49.459→102.510（+107.26%）、Z为3.072→5.451、Yaw为26.914→43.329；这三项十图全部变差。到19550终点，XY仍为95.145，对照α4为47.050，四项定位指标十图均较差。final的α12 FVD更低（879.816 vs921.205），但连续SSIM／OFE、离散SSIM／FID更差。这里是表现退化证据，不凭最终指标判定训练发散、梯度冲突或具体根因。
3. **Stage 2 α12与Stage 1 α12必须分开判断。** 固定exp37@12000来源、同local4000，α12相对α8的XY为52.309→50.624（−3.22%）、Pitch为3.236→2.667（−17.58%）、FVD为973.564→937.934（−3.66%），三项均十图改善；但Z／Yaw、离散LPIPS／Boundary_F1和部分时序误差变差。α12并未重现从头训练时的严重定位退化，说明权重效果取决于训练阶段与已有模型状态，仍需固定7550终点判断。
4. **提前切换到低LR Stage 2并不自动改善三任务。** 同累计16000、同α8，exp38_s2_joint相对exp37的XY／Z／Yaw更好、Pitch更差；离散仅Boundary_F1改善，连续八项宏指标全部较差，FVD为969.281→973.564。两路线差异包含LR路径、优化器／scheduler重置及warmup，不能只归因于联合任务或辅助loss。
5. **both当前是跨batch观察结果，尚不足以证明辅助损失有效。** 相对joint，both的XY为52.309→54.076、Z为2.974→3.248，Pitch为3.236→2.954；FID为37.218→37.043、FVD为973.564→965.626。定位有取舍、生成差异较小，且推理batch16／8与32／16不同，不能将这组差值直接记为辅助收益或辅助损害。尤其LPIPS等三位小数接近的差异不应被夸大。

以上改善数与百分比使用未舍入的十图宏平均；逐图一致性描述覆盖范围，不把它当成独立重复实验。Stage 1／Stage 2分别比较，累计步数相同不等于FLOPs或GPU时间相同。

### 后续工作顺序

1. 先补齐both与joint的相同推理batch对照，再评价P4＋U2增量；同步已有结果即可，缺项再由用户执行。exp37@16000三任务现已齐，无需重复本节点推理或评测。
2. 完成／同步exp38四分支固定local7550，以及exp37@19550的batch32／16对照；保留两组exp39的固定final，不用test选择不同指标的最佳checkpoint。
3. 补齐既定validation并核对Stage 1 α12定位训练曲线和预测误差分布，再决定后续配方。当前α4是更均衡的Stage 1候选；Stage 2 α12值得按既定终点继续观察。需要新辅助实验时，仍应从同一基座成对设置无辅助／有辅助，不能直接拿exp39与exp38的跨阶段差异当作辅助消融。

本次仅整理已有产物并更新文档；未启动训练、推理或评测，未修改代码、配置或record命令。

<a id="constloc4-stage1-stage2-20261109"></a>

## 2026-11-09：Stage 1／Stage 2 独立 constloc4 对照

同时准备两个独立实验，分别检验前期学习和后期继续优化的主任务权重。用户手动安排运行；本次不启动训练、推理或评测。`exp38_s2_joint_constloc4` 与 `exp39` 不相互依赖，可以在不同GPU上并行。它们都关闭 perception、aux-loc 和其他辅助损失。

### 设计与对照关系

| 实验 | 初始模型 | Stage 1 αloc／步数 | Stage 2 αloc／步数 | 保存checkpoint | 累计更新数 | 主要对照 |
|---|---|---|---|---|---|---|
| `exp37`（已有） | 原始UniLIP-1B＋π0.5 action head | 8／19550 | 无 | 已有12000、16000、19550 | 19550 | Stage 1 α=8 |
| `exp39`（新增） | 同exp37的原始初始化；不加载exp37 | 4／19550 | 无 | **12000、16000、19550** | 19550 | 同step的exp37 |
| `exp38_s2_joint`（已有） | exp37@12000，仅模型权重 | 8／12000 | 8／7550 | local4000、7550 | 16000、19550 | Stage 2 α=8 |
| `exp38_s2_joint_constloc4`（新增） | **同一exp37@12000**，固定来源SHA-256 | 8／12000 | **4／7550** | **local4000、7550** | **16000、19550** | 同local step的exp38_s2_joint |

- Stage 2效应：`exp38_s2_joint_constloc4` 对照 `exp38_s2_joint`，只改变继续训练阶段的 `alpha_loc_loss: 8→4`；来源、优化器重置、LR、训练长度、保存节点和数据均一致。
- Stage 1效应：`exp39@16000/final` 对照 `exp37@16000/final`，从同一原始初始化开始，主定位权重全程4或8。exp39固定19550更新，避免设备布局改变epoch推导步数时造成预算差异；保存策略属于产物管理差异。
- `exp39@12000` 为未来Stage 2保留来源。本轮只建立这两个新实验，不自动创建以exp39为来源的Stage 2。现有exp38 runner仍严格要求exp37@12000，不能直接替换路径后冒用exp38实验身份。
- `exp39@19550` 与两个exp38 joint终点都是19550累计更新，但跨阶段比较还包含优化器重置和LR路径差异，不能只归因于α。旧exp32单任务final19500比它们少50更新；16000节点则更新数一致。所有“等预算”仅指optimizer更新次数，不表示等FLOPs。

### 固定训练超参

| 项目 | `exp38_s2_joint_constloc4` | `exp39` |
|---|---|---|
| 入口／配置 | `scripts/run_exp38_seen10.py`／`csgo_configs/exp38_s2_joint_constloc4.yaml` | **复用** `scripts/run_exp37_seen10.py`／`csgo_configs/exp39.yaml` |
| 总训练horizon | local7550，来源已经12000 | **19550**，从原始初始化开始 |
| 主定位权重 | 恒定4，无主权重ramp | 恒定4，无主权重ramp |
| 其他loss | 生成主损失系数1；全部辅助关闭 | 相同 |
| 普通参数／LLM、生成头、定位头LoRA／action IO峰值LR | 2e-5 | 1e-4 |
| action connector／norm／projector峰值LR | 1e-4 | 5e-4 |
| LR warmup | 252步 | ratio=0.003，对19550步为59步 |
| LR scheduler | cosine_with_min_lr，min_lr_rate=0.1 | 沿用exp37的cosine_with_min_lr，min_lr=1e-5及参数组比例衰减 |
| optimizer | 新AdamW，β=(0.9,0.999)，eps=1e-8，weight_decay=0，clip=1 | 相同；没有来源optimizer |
| LoRA | r32／alpha64／dropout0.05，模块开关沿用父配方 | 相同 |
| batch／数据 | 每步128基础三元组，balanced paired，每个样本同时有gen和loc；Seen-10 50000训练样本；seed42 | 相同 |
| 保存／保留 | 精确4000、7550，完整恢复状态，keep2 | 精确12000、16000、19550，完整恢复状态，**keep3** |
| 根目录完整模型副本 | 不导出 | 不导出；final指向checkpoint-19550 |

exp39的 `max_steps=19550` 始终固定，`num_train_epochs=50` 沿用原配置语义；不能为生成12000来源把max_steps改为12000或压缩cosine。支持 `--stop-after-step 12000/16000/19550`；`--stop-after-step 0` 连续运行到19550。暂停后使用本实验 `--resume` 恢复完整optimizer／scheduler／RNG，不重新warmup。exp39默认连续跑完整horizon；exp38新组沿用默认首停4000，record明确提供一次跑完7550的命令。

两份配置默认使用minimal资产清单；首次train使用 `--asset-mode full --data-dir /path/to/preprocessed_data` 可切换full。后续resume及评测继承保存的resolved_config；不在同一次实验中切换资产后端。microbatch与GPU数可在首次运行时选择，GAS自动维持基础batch128；同一实验恢复时保持原布局。

### checkpoint用途与评测安排

| 模型节点 | 用途 | validation | test |
|---|---|---|---|
| exp39@12000 | 保存未来Stage 2模型来源 | 可选支持定位＋离散，不列入默认队列 | 可选支持三任务，不列入默认队列 |
| exp39@16000 | 与exp37@16000及exp38 local4000做等更新次数对照 | 定位＋离散生成 | 定位＋离散＋连续生成 |
| exp39@19550／final | 完整Stage 1终点，对比exp37 final及exp38 final | 定位＋离散生成 | 定位＋离散＋连续生成 |
| exp38新组local4000 | 对比原joint@4000，累计16000 | 沿用exp38协议，不新增 | 定位＋离散＋连续生成 |
| exp38新组local7550／final | 对比原joint@7550，累计19550 | 沿用exp38协议，不新增 | 定位＋离散＋连续生成 |

新增exp37@16000的validation／test路由，用于上述Stage 1对照；保留exp37旧训练runtime、恢复指纹、其他Stage2来源及评测行为。推理统一seed42、定位batch32、生成batch16，metrics使用相同batch参数定位结果目录。`--task all`在test包含三任务，在validation只包含定位与离散生成。test@16000和final是预先指定的报告点，不用test挑选阶段、权重或各指标最优checkpoint。exp37已有不同batch的final结果保持原样；record提供同batch对照命令，已有相同产物时可只补缺项。

### 实施与进度

以下保留新增实验的设计，进度更新至2026-11-12；实测指标见上方结果表。所有checkpoint输出均为独立实验目录，不复制或覆盖exp37权重。

| 实验 | 训练状态 | checkpoint保留 | validation状态 | test状态 |
|---|---|---|---|---|
| `exp38_s2_joint_constloc4` | 已训练至local4000/7550；模型／状态已同步 | local4000已到；7550待同步 | 不新增 | local4000三任务完成；7550未见同步结果 |
| `exp39` | 已完成19550/19550 | 12000／16000／19550模型及状态已同步 | 两节点未见同步结果 | 16000／19550三任务均完成，batch32／16 |
| `exp37`（对照） | 已有完整权重 | 本机有12000／16000／19550 | 16000待补；final已有历史结果 | 16000三任务推理／评测均完成；final同batch32／16待补 |

可执行命令见 [record.md新增Stage1／Stage2权重对照](record.md#constloc4-stage1-stage2-20261109)。代码、配置和手动命令已完成；配置及测试通过不代表GPU训练或评测完成。

本轮验收：exp37／exp39入口32项、exp38入口15项、精确保存与恢复8项、Unseen-4入口13项，共68项CPU测试通过。24组实际CLI的validate／只读train、infer、metrics计划通过，覆盖新增两组及exp37@16000／final对照；没有创建新增实验的训练、推理或评测输出目录。另对本机exp37、exp37_s2、exp37_s2_gen、exp37_s2_loc、exp37_s2_both五份已保存配方重算指纹，均与原训练计划一致。两份新配置与各自父配方的结构化差异检查通过，修改范围仅为主权重、实验身份及明确列出的checkpoint控制。

record新增6段Bash命令及其中2段nohup子shell正文均通过 `bash -n`；保留record原有CRLF。独立检查未发现新增实验输出目录，未执行项目训练、推理或指标计算。

<a id="constloc12-stage1-stage2-20261109"></a>

## 2026-11-09：补充 Stage 1／Stage 2 constloc12 对照

新增 `exp38_s2_joint_constloc12` 和 `exp39_consloc12`；后者按用户指定保留 **consloc12** 拼写，配置、CLI、输出目录与命令一致。两份配置分别继承 `exp38_s2_joint_constloc4` 和 `exp39`，仅将 `alpha_loc_loss: 4.0` 改为 `12.0`，并修改实验身份字段。父实验配置和既有结果保留；未启动实验。

| 实验 | 继承配置 | 模型初始化 | 主定位权重 | 训练horizon／保存节点 | 主要对照 |
|---|---|---|---|---|---|
| `exp38_s2_joint_constloc12` | `exp38_s2_joint_constloc4` | **exp37@12000，来源αloc=8**，仅加载模型 | Stage 2恒定12 | local7550；保留4000、7550，累计16000、19550 | 同源Stage 2 αloc=4／8／12 |
| `exp39_consloc12` | `exp39` | **原始UniLIP-1B＋π0.5**；不加载exp39或exp37 | 全程恒定12 | 固定19550；保留12000、16000、19550 | Stage 1的exp39(4)／exp37(8)／本组(12) |

这里的“继承”指配置配方，不表示继承父实验训练后的模型。Stage 2来源继续固定为exp37@12000；新Stage 1从原始初始化开始。两组无perception／aux-loc及其他辅助loss，不添加主权重ramp。

- Stage 2其余超参保持普通LR=2e-5、action connector／norm／projector LR=1e-4、warmup252、cosine最低比例0.1、新AdamW、keep2、关闭根模型重复导出；默认首停4000，恢复到7550的调度horizon不变。
- Stage 1其余超参保持普通LR=1e-4、action connector／norm／projector LR=5e-4、warmup ratio=0.003、exp39原cosine配置、新AdamW、keep3、关闭根模型重复导出。完整训练会保留12000，不必提前暂停才能保留该来源；只有希望在12000暂停时使用 `--stop-after-step 12000`。
- 两组沿用paired joint、有效基础batch128、LoRA r32／alpha64／dropout0.05、seed42、原数据划分及minimal默认资产；首次train可用 `--asset-mode full --data-dir PATH` 切换full，resume及评测继承保存的资产设置。GPU布局与microbatch在本实验恢复时保持一致。
- Stage 2沿用 `scripts/run_exp38_seen10.py`，test节点为4000／7550；Stage 1沿用 `scripts/run_exp37_seen10.py`，支持12000／16000／final的validation及test，默认报告16000／final，12000主要保留为未来Stage 2来源。test包括定位／离散／连续，validation只包括定位／离散。推理统一定位batch32、生成batch16、seed42，metrics使用相同参数定位产物。
- 两组使用独立输出目录及恢复身份，只能 `--resume` 自己的完整checkpoint；不能用α4实验的optimizer状态继续训练冒充新组。未来从 `exp39_consloc12@12000` 出发的Stage 2不在本轮创建，现有exp38来源限制不放宽。

| 实验 | 训练状态 | checkpoint保留 | validation状态 | test状态 |
|---|---|---|---|---|
| `exp38_s2_joint_constloc12` | 本机状态4000/7550；4000模型／计划／状态已同步 | 4000已到；7550未见 | 不新增 | local4000三任务完成，batch32／16；7550未见结果 |
| `exp39_consloc12` | 已完成19550/19550 | 12000／16000／19550模型及状态已同步 | 两节点未见同步结果 | 16000／19550三任务均完成，batch32／16 |

新增配置：[Stage 2 αloc12](csgo_configs/exp38_s2_joint_constloc12.yaml)、[Stage 1 αloc12](csgo_configs/exp39_consloc12.yaml)。可直接执行的训练／推理／评测及恢复命令见 [record.md](record.md#constloc12-stage1-stage2-20261109)。固定step对照使用完整指标，不以中间test选择权重或拼接最优checkpoint；相同累计更新数不代表相同计算量。

本轮验收：两个runner的54项CPU测试通过；两份配置与父配置的结构化差异严格只有主定位权重4→12和实验身份。18组CLI validate／只读train、infer、metrics计划通过，覆盖minimal／full与报告节点；新增6段Bash及3段nohup子shell正文通过语法检查。相对本轮变更前代码，原有10组实验的训练命令、runtime和fingerprint保持一致。record原内容逐字节保留，仅追加新命令；没有创建这两组的训练／推理／评测输出目录，未启动实验。
