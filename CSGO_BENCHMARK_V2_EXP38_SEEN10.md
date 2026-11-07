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
| checkpoint／test节点 | local3000、7550（final） | 相同 |
| 对应累计节点 | 15000、19550 | 相同 |

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

新runner固定 `max_steps=7550`。在local3000暂停不会把scheduler改成3000步；恢复后继续原7550步调度，不重新warmup、不重新应用来源模型初始化。命令中的epoch上限不表示Stage 2额外训练50轮。

### 数据与评测

沿用完整Seen-10：训练50000条基础三元组；test定位／离散生成各20000条，连续生成200×64=12800帧，十图等权宏平均。默认minimal资产；首次训练可切换为full资产，后续resume／infer／metrics使用本实验保存的资产后端与路径。

两组、两个节点均做test定位、离散生成和连续生成。默认固定 `--test-loc-batch-size 32`、`--test-gen-batch-size 16`；metrics传入相同值以选中同一推理目录。支持分别执行 `localization`、`gen`、`gen_continuous`，`all`包含三项。定位指标由infer生成，metrics校验其摘要；生成metrics逐图计算并汇总。连续生成沿用逐帧条件生成，不增加历史帧输入。

local3000是预先指定的中间test报告点，local7550是固定正式终点；不根据3000的test结果调参、早停、选择checkpoint或拼接不同节点的最佳指标。本轮不新增validation节点。生成Dropout与批量行为沿用现有推理实现；新两臂可在相同batch下比较，与历史batch1／8结果对照时保留推理设置差异。

### 保存、暂停与恢复

完整模型及恢复状态**只保存 `checkpoint-3000/` 和 `checkpoint-7550/`**，每份包含模型、optimizer／scheduler、各rank RNG和Trainer状态。不能采用周期3000保存，否则会额外产生6000且可能挤掉3000。新配置显式设 `training_checkpoint_steps: [3000,7550]`、`save_final_model_at_output_root: false`；精确节点callback负责保存，`save_total_limit=2`。保留根目录配置、训练计划、轻量Trainer状态及日志，但不额外导出一份根目录完整模型／adapter。

推荐手动串行流程：

1. 首次训练到local3000后保存并停止（全程horizon仍为7550）。
2. 对3000执行test推理及评测。
3. 本实验 `--resume --stop-after-step 0`，从3000的完整状态继续到7550。
4. 对final=7550执行test推理及评测。

直接一次训练到7550时也必须仅保存3000、7550，之后可分别补两节点评测。没有更早的恢复checkpoint；3000后的中断可从3000续训。首次不能加 `--resume`，也不能从另一个exp38实验的checkpoint恢复。runner仅接受这两个评测节点，拒绝其他step及validation，不删除其他实验产物。各节点的三任务结果、manifest及评测配置保留在对应实验／checkpoint／test／seed／batch目录。

### 实验定位与对照

- **辅助增量**：同一local节点的 `exp38_s2_both` 对照 `exp38_s2_joint`。
- **等累计更新次数的两阶段方案**：final两组对照 `exp37@19550`；变化包含提前切阶段、LR路径与优化器重置，不能全部归因于辅助损失。
- 旧单任务 `exp32_gen/loc@19500` 比本方案少50更新（约0.256%），只能称近似更新预算。联合与辅助目标每步计算量不同，**不宣称等FLOPs或等GPU时间**。
- 相比原联合两阶段路线19550＋6000＝25550，新路线减少6000累计更新；复用已有12000权重时，此次每臂新增训练7550步，两臂合计15100步。论文方法成本仍包含12000步前缀，不能把共享前缀计为零。

## 变更方案

1. 新增独立入口 `scripts/run_exp38_seen10.py` 和两份同名训练YAML，复用现有安全工具／评测接口；保留exp37及Unseen-4的既有来源、终点和任务约束。
2. 来源预检仅接受exp37@12000且原horizon=19550，核对已保存recipe／plan、十图协议、模型结构、LoRA header和SHA-256；两臂固定相同来源。源资产布局不限制新运行选择minimal或full，但数据协议必须相同。
3. 在 `unilip/train/csgo_staged_training.py` 与 `train_csgo.py` 增加可选的精确保存列表及关闭根模型重复导出逻辑；未设置新字段时沿用旧行为。
4. 新runner提供validate／train／infer／metrics、默认只打印计划；只有用户手动传 `--execute` 才写实验快照并执行。3000检查保存状态的 `global_step=3000,max_steps=7550`，final要求7550/7550。
5. `record.md` 分别提供两组首次到3000、test、恢复到final、final test的直接执行命令，以及minimal／full切换、Slurm中断恢复说明；不会自动启动后台任务。
6. CPU检查覆盖来源错误拒绝、7550 horizon、252／503调度、两节点保存与保留、根模型跳过、3000暂停恢复、三任务批量路由、minimal／full及只读dry run。既有exp37／Unseen入口与默认保存行为执行必要回归。

## 实施与实验进度

设计先于实现保存；代码、配置和手动命令已完成，并按用户最新指令统一采用 **P4＋U2、perception attention=false**。未启动实验。

- 新入口：[run_exp38_seen10.py](scripts/run_exp38_seen10.py)；配置：[joint](csgo_configs/exp38_s2_joint.yaml)、[both](csgo_configs/exp38_s2_both.yaml)。`validate` 和不带 `--execute` 的命令只读；首次train默认在3000停下，`--stop-after-step 0` 可一次跑完7550。
- 新runner的9项CPU测试通过；精确保存／暂停恢复的8项CPU测试通过，使用临时目录中的合成小模型确认同一horizon、两节点完整恢复材料、跨epoch无额外checkpoint及不重复导出根模型。既有exp37入口23项、Unseen-4入口13项回归通过，共53项。
- 16组训练计划＋64组infer／metrics计划核验通过，覆盖minimal／full、单卡／双卡、两个test节点、三项任务和默认batch32／16；验证未创建任何exp38的训练／推理／评测输出目录。两份nohup脚本通过 `bash -n`，保留record.md原有CRLF。
- 对真实来源进行了只读预检：12000/19550状态、2830个张量／1128个LoRA张量、模型文件4,577,302,678字节；SHA-256为 `0388328dc2e260bca15bb071fc4e9ff189fe631686fba553f187aea9917de8ae`。未将大模型载入GPU或执行项目训练／推理／评测；首次手动执行时runner仍会重新预检并将来源绑定到本实验计划。
- 新保存字段未配置时保留原训练保存行为；exp37／Unseen runner未修改。真实大模型训练吞吐、显存与最终指标等待用户手动运行后记录。

| 实验 | 训练 | test@3000（累计15000） | test@7550（累计19550） |
|---|---|---|---|
| `exp38_s2_joint` | 未启动；计划local7550 | 待执行：定位／离散／连续 | 待执行：定位／离散／连续 |
| `exp38_s2_both` | 未启动；计划local7550 | 待执行：定位／离散／连续 | 待执行：定位／离散／连续 |

上表为尚未运行的实验计划，不填写实测指标。执行命令见 [record.md](record.md#exp38-seen10-20261107)。
