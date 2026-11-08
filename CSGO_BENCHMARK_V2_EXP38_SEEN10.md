# CSGO Benchmark v2：exp38 Seen-10 固定总更新预算的两阶段实验

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

设计先于实现保存；代码、配置和手动命令已完成，并按用户最新指令统一采用 **P4＋U2、perception attention=false**。未启动实验。

- 新入口：[run_exp38_seen10.py](scripts/run_exp38_seen10.py)；配置：[joint](csgo_configs/exp38_s2_joint.yaml)、[both](csgo_configs/exp38_s2_both.yaml)。`validate` 和不带 `--execute` 的命令只读；首次train默认在4000停下，`--stop-after-step 0` 可一次跑完7550。
- 新runner的9项CPU测试通过；精确保存／暂停恢复的8项CPU测试通过，使用临时目录中的合成小模型确认同一horizon、两节点完整恢复材料、跨epoch无额外checkpoint及不重复导出根模型。既有exp37入口23项、Unseen-4入口13项回归通过，共53项。
- 16组训练计划＋64组infer／metrics计划核验通过，覆盖minimal／full、单卡／双卡、两个test节点、三项任务和默认batch32／16；验证未创建任何exp38的训练／推理／评测输出目录。两份nohup脚本通过 `bash -n`，保留record.md原有CRLF。
- 对真实来源进行了只读预检：12000/19550状态、2830个张量／1128个LoRA张量、模型文件4,577,302,678字节；SHA-256为 `0388328dc2e260bca15bb071fc4e9ff189fe631686fba553f187aea9917de8ae`。未将大模型载入GPU或执行项目训练／推理／评测；首次手动执行时runner仍会重新预检并将来源绑定到本实验计划。
- 新保存字段未配置时保留原训练保存行为；exp37／Unseen runner未修改。真实大模型训练吞吐、显存与最终指标等待用户手动运行后记录。

| 实验 | 训练 | test@4000（累计16000） | test@7550（累计19550） |
|---|---|---|---|
| `exp38_s2_joint` | 未启动；计划local7550 | 待执行：定位／离散／连续 | 待执行：定位／离散／连续 |
| `exp38_s2_both` | 未启动；计划local7550 | 待执行：定位／离散／连续 | 待执行：定位／离散／连续 |

上表为尚未运行的实验计划，不填写实测指标。执行命令见 [record.md](record.md#exp38-seen10-20261107)。

## 中间节点对照：exp32 单任务 @16000（2026-11-08 更新）

按指定对照节点，已同步并核对 `exp32_loc@16000` 的 Seen-10 test 定位和 `exp32_gen@16000` 的 test 离散／连续生成结果，**三个任务均已完成**。推理manifest与结果摘要均绑定各自的 `checkpoint-16000/model.safetensors`；两份checkpoint的 `trainer_state.json` 均为 `global_step=16000,max_steps=19500`。定位／离散各20000条，连续12800帧，均覆盖完整十图。

历史19500步最终结果继续保留；下表区分其身份，本节16000对照表使用新同步的独立结果目录：

| 已有结果 | manifest／summary 绑定的权重 | 已有 test 范围 | 本次处理 |
|---|---|---|---|
| `outputs_loc/benchmark_v2/exp32_loc/seen/benchmark_csgo_v2_loc.json` | `outputs/csgo_1b/exp32_loc/model.safetensors`，根目录状态为19500 | `seen_discrete_test`，10图／20000条 | 保留原最终结果；16000定位已另行补齐 |
| `outputs_eval/benchmark_v2/exp32_gen/seen/discrete/summary.json` | `outputs/csgo_1b/exp32_gen/model.safetensors`，根目录状态为19500 | `seen_discrete_test`，10图／20000条 | 保留原最终结果；16000离散生成已另行补齐 |
| `outputs_eval/benchmark_v2/exp32_gen/seen/continuous/summary.json` | 同上 | `seen_continuous`，10图／12800帧 | 保留原最终结果；16000连续生成已另行补齐 |

**比较口径（2026-11-09 更正累计步数）**：exp38 的 Stage 2 local4000 对应累计 `12000+4000=16000` 次更新，与 exp32 单任务16000、exp37／exp39单阶段16000的更新次数相同。原文15000及多1000次更新的说明是算术笔误；不涉及checkpoint或训练配置变更。相同更新次数不等于相同FLOPs或GPU时间。辅助收益仍首先由同节点的 `exp38_s2_both` 与 `exp38_s2_joint` 判断，不根据中间test结果选择checkpoint。

本次同步结果使用 seed42、定位batch32、生成batch16，与exp38计划一致；资产后端为minimal，generation metrics使用 `benchmark_v2_core`、指标batch1、paired_size448、逐图评测后十图等权宏平均。连续生成沿用clip_length16／clip_stride16／fvd_size224。定位推理脚本同时输出定位评测，无需再执行一轮定位推理来计算指标。完整可执行命令及full资产切换说明见 [exp32_gen@16000](record.md#exp32-gen-test-16000) 和 [exp32_loc@16000](record.md#exp32-loc-test-16000)。

已核对十图逐图结果与摘要的宏平均值一致；生成逐图预测覆盖率均为100%，无缺失或额外预测。实测来源如下：

- 定位：[benchmark_csgo_v2_loc.json](outputs_loc/benchmark_v2/exp32_loc/checkpoint-16000/test/seed_42/infer_bs32/benchmark_csgo_v2_loc.json)。
- 离散生成：[discrete/summary.json](outputs_eval/benchmark_v2/exp32_gen/checkpoint-16000/test/seed_42/infer_bs16/discrete/summary.json)。
- 连续生成：[continuous/summary.json](outputs_eval/benchmark_v2/exp32_gen/checkpoint-16000/test/seed_42/infer_bs16/continuous/summary.json)。

以下三表的exp32单任务行已填入16000步实测宏平均结果，统一保留三位小数；exp38尚未同步结果，继续预留其local4000对照行。“—”表示尚无实测结果。本次仅整理已有产物，未启动训练、推理或评测。

### test 定位对照

| 实验 | checkpoint | 累计训练step | 状态 | XY↓ | Z↓ | Pitch↓ | Yaw↓ |
|---|---|---|---|---|---|---|---|
| `exp32_loc` | 16000（单阶段） | 16000 | test推理／评测已完成 | 50.708 | 3.050 | 3.039 | 26.880 |
| `exp38_s2_joint` | Stage 2 local4000 | 16000 | 待训练／推理／评测 | — | — | — | — |
| `exp38_s2_both` | Stage 2 local4000 | 16000 | 待训练／推理／评测 | — | — | — | — |
| `exp38_s2_joint_constloc4` | Stage 2 local4000 | 16000 | 新配置；未启动 | — | — | — | — |
| `exp39` | 16000（单阶段，αloc=4） | 16000 | 新配置；未启动 | — | — | — | — |
| `exp37` | 16000（单阶段，αloc=8） | 16000 | checkpoint已有；test待补 | — | — | — | — |

### test 离散生成对照

| 实验 | checkpoint | 累计训练step | 状态 | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ |
|---|---|---|---|---|---|---|---|---|
| `exp32_gen` | 16000（单阶段） | 16000 | test推理／评测已完成 | 13.815 | 0.410 | 0.616 | 0.514 | 36.167 |
| `exp38_s2_joint` | Stage 2 local4000 | 16000 | 待训练／推理／评测 | — | — | — | — | — |
| `exp38_s2_both` | Stage 2 local4000 | 16000 | 待训练／推理／评测 | — | — | — | — | — |
| `exp38_s2_joint_constloc4` | Stage 2 local4000 | 16000 | 新配置；未启动 | — | — | — | — | — |
| `exp39` | 16000（单阶段，αloc=4） | 16000 | 新配置；未启动 | — | — | — | — | — |
| `exp37` | 16000（单阶段，αloc=8） | 16000 | checkpoint已有；test待补 | — | — | — | — | — |

### test 连续生成对照

| 实验 | checkpoint | 累计训练step | 状态 | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | TWE↓ | TDE↓ | OFE↓ | FVD↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `exp32_gen` | 16000（单阶段） | 16000 | test推理／评测已完成 | 14.098 | 0.412 | 0.605 | 0.523 | 36.496 | 42.628 | 17.365 | 867.691 |
| `exp38_s2_joint` | Stage 2 local4000 | 16000 | 待训练／推理／评测 | — | — | — | — | — | — | — | — |
| `exp38_s2_both` | Stage 2 local4000 | 16000 | 待训练／推理／评测 | — | — | — | — | — | — | — | — |
| `exp38_s2_joint_constloc4` | Stage 2 local4000 | 16000 | 新配置；未启动 | — | — | — | — | — | — | — | — |
| `exp39` | 16000（单阶段，αloc=4） | 16000 | 新配置；未启动 | — | — | — | — | — | — | — | — |
| `exp37` | 16000（单阶段，αloc=8） | 16000 | checkpoint已有；test待补 | — | — | — | — | — | — | — | — |

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

以下为新增实验计划，未填写实测指标。所有checkpoint输出均为独立实验目录，不复制或覆盖exp37权重。

| 实验 | 训练状态 | checkpoint保留 | validation状态 | test状态 |
|---|---|---|---|---|
| `exp38_s2_joint_constloc4` | 新配置；未启动 | 计划local4000／7550 | 不新增 | 两节点三任务待执行 |
| `exp39` | 新配置；未启动 | 计划12000／16000／19550 | 16000／19550待执行 | 16000／19550三任务待执行 |
| `exp37`（对照） | 已有完整权重 | 本机有12000／16000／19550 | 16000待补；final已有历史结果 | 16000待补；final按同batch核对或补齐 |

可执行命令见 [record.md新增Stage1／Stage2权重对照](record.md#constloc4-stage1-stage2-20261109)。代码、配置和手动命令已完成；配置及测试通过不代表GPU训练或评测完成。

本轮验收：exp37／exp39入口32项、exp38入口15项、精确保存与恢复8项、Unseen-4入口13项，共68项CPU测试通过。24组实际CLI的validate／只读train、infer、metrics计划通过，覆盖新增两组及exp37@16000／final对照；没有创建新增实验的训练、推理或评测输出目录。另对本机exp37、exp37_s2、exp37_s2_gen、exp37_s2_loc、exp37_s2_both五份已保存配方重算指纹，均与原训练计划一致。两份新配置与各自父配方的结构化差异检查通过，修改范围仅为主权重、实验身份及明确列出的checkpoint控制。

record新增6段Bash命令及其中2段nohup子shell正文均通过 `bash -n`；保留record原有CRLF。独立检查未发现新增实验输出目录，未执行项目训练、推理或指标计算。
