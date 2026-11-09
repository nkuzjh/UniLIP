# CSGO Benchmark v2：exp37 Unseen-4 每图独立 few-shot

状态（2026-11-09 本机同步核查）：**`exp37_fs_gen`／`exp37_fs_loc` 各16组与 `exp37_fs_both` 四图四档16组，共48个final400训练终点已独立核验**。both 四图四档共16/16点的定位、离散、连续query指标全部齐全（48/48份）；正式训练权重与Trainer状态均已同步并核验400/400。both 定位 infer_bs16、生成 infer_bs32；zero-shot三项齐全。旧 gen／loc／joint 为 batch=1；跨batch结果不可严格归因辅助loss。joint／both_b1 few-shot 仍无本机完成证据；retention暂停。协议、运行入口和手动命令见后文。

## 实验设计定位

每个来源先在四张 Unseen 地图做 zero-shot；再对每张地图、每个 shot 独立创建一个模型。四图固定为 `cs_office`、`de_golden`、`de_palacio`、`de_vertigo`。这里只使用其 manifest 冻结的 support，不使用地图全部采集数据。

| Few-shot family | 直接来源 | 任务 | 固定 αloc | 固定 aux-loc | 固定 perception | attention |
|---|---|---|---:|---:|---:|---|
| `exp37_fs_gen` | `exp37_s2_gen@6000` | gen | 不参与 | 0 | 0 | 不参与 |
| `exp37_fs_loc` | `exp37_s2_loc@6000` | loc | 1 | 0 | 0 | 不参与 |
| `exp37_fs_joint` | `exp37_s2@6000` | joint | 8 | 0 | 0 | 不参与 |
| `exp37_fs_both` | `exp37_s2_both@6000` | joint，P4＋U2 | 8 | 1 | 0.05 | false |
| `exp37_fs_both_b1` | `exp37_s2_both_b1@6000` | joint，P3＋U2 | 8 | 1 | 0.05 | true |

适配模型名为 `<family>_<map>`，例如 `exp37_fs_both_cs_office`。shot 和 support seed 放在输出子目录，不创建另外的模型名。五个 family × 四图 × 四种 shot = **80 次独立适配训练**，总计 32,000 optimizer updates；zero-shot 使用五个原模型，不增加训练。若在查看 Unseen 结果前依据 Seen validation 只选择一个 both 候选，则实际运行四个 family、64 次适配即可。代码同时保留两个候选，不自动择优、不自动运行矩阵。

各来源都必须真正完成 Seen Stage2 local6000。不能以 Stage1 `exp37@19550`、Stage2@4000、旧 `exp32` 或另一张地图的适配模型代替。只加载模型权重，新建 optimizer、scheduler 和 local step；100/50/20/10-shot 之间不能接续训练。现有单任务 Stage2 源自旧 `exp32_gen/loc`，联合 Stage2 源自新 `exp37`，应保留这一区别，不能将整个历史训练预算声称为 FLOPs 相同。

## 公共训练设置

| 参数 | 固定设置 |
|---|---|
| manifest | `data/csgo_benchmark_v2/benchmark_manifest.json` |
| split／地图 | `crossmap_support`，仅本次目标地图 |
| support | `support_seed=0`；`100/50/20/10`，嵌套 `10⊂20⊂50⊂100` |
| 训练／推理随机种子 | 42；与 support seed=0 分开记录 |
| 训练长度 | `--max_steps 400`，以 optimizer update 为单位；epoch 不决定终点 |
| joint batch | 单卡 microbatch=4、gradient accumulation=32；paired joint，每个 base sample 同时提供 loc 和 gen |
| 单任务 batch | 单卡 microbatch=SHOTS、gradient accumulation=1 |
| 普通参数／活动 LoRA 峰值 LR | `2e-5` |
| Action connector／norm／projector 峰值 LR | `1e-4`，仅活动定位模块 |
| optimizer | AdamW，β=(0.9,0.999)，eps=1e-8，weight decay=0，clip norm=1 |
| scheduler | `cosine_with_min_lr`，`min_lr_rate=0.1`；普通组最低2e-6，Action上述组最低1e-5 |
| LR warmup | `warmup_ratio=0.003`，400步约2步；清除 Seen Stage2 的 warmup_steps=200 |
| 主／辅助权重 | 使用上表成熟权重，从 local0 固定；不重放任何 loss ramp |
| U2 细节 | combined，候选数2，candidate_tau=1，unc_tau=1，unc_min_weight=0.05，共享定位噪声；exp_sigma λ=5 |
| perception 细节 | 当前定位头、teacher detach、无EMA；vision token SmoothL1，linear_1m_sigma；保留P3/P4 attention差异 |
| trainability | 继承各自 Stage2 的共享／私有 LoRA 和冻结设置；单任务冻结非活动头；不改变组件实际 LoRA rank |
| 精度／显存 | bf16、tf32、gradient checkpointing，沿用项目实现 |
| checkpoint／验证 | 不周期保存，最终 local400 保存根目录模型和Trainer状态；无 Unseen 中途验证或选点 |

短适配的学习率是本次方案：普通组沿用 exp37 Stage2 的2e-5，warmup长度沿用原 few-shot 比例。它不同于旧 exp35/36 的基础学习率1e-4；不修改旧配置，也不将新旧模型差异完全归因为辅助损失。

### 短数据集 batch 与 sampler

原任务 sampler 只发出完整微批，Trainer 在 epoch 尾执行不足32个微批的累积更新。因此 joint 每个 optimizer step 的实际 base sample 数如下，**名义4×32不能写成实际128**：

| 每图 shot | 每epoch微批数 | 每update实际base samples | 400步base sample曝光 | 400步loc＋gen任务实例 |
|---:|---:|---:|---:|---:|
| 100 | 25 | 100 | 40000 | 80000 |
| 50 | 12 | 48 | 19200 | 38400 |
| 20 | 5 | 20 | 8000 | 16000 |
| 10 | 2 | 8 | 3200 | 6400 |

`csgo_loss_per_microbatch_mean=true` 保留按实际累积微批数归一化。单任务每步使用全部SHOTS个support，400步曝光分别40000/20000/8000/4000；协议固定更新次数，未使联合与单任务总曝光完全相等。

本次 CPU 索引检查发现旧裸 DataLoader 不转发 `set_epoch`，50/10-shot 会在每epoch固定遗漏同两条support。新增 `csgo_forward_sampler_epoch=true` **仅在本次适配启用**，将 Trainer 的 epoch 传给 batch sampler，使被丢弃的尾样本随epoch轮换；所有原配置默认行为保持不变。它不改变冻结support集合、microbatch、累积数或400步预算。此行为差异应随新结果记录，不能把旧曲线当成采用了同一修复的结果。

## 推理与评测设置

| 模式 | checkpoint | 地图／split | 定位／离散生成数量 | 连续生成数量 |
|---|---|---|---:|---:|
| zero-shot | 对应Seen Stage2@6000，未适配 | 全Unseen-4，query／continuous | 各8000 | 80×64=5120帧 |
| few-shot query | 本次适配final local400 | 单一目标地图，query／continuous | 各2000 | 20×64=1280帧 |
| Seen retention | 同一适配final local400 | 全Seen-10，test／continuous | 各20000 | 200×64=12800帧 |

`--task all` 根据来源路由：joint=定位＋离散生成＋连续生成，gen=两类生成，loc=仅定位。`--task loc/gen/gen_continuous` 可单独执行一项，方便在远程作业中断后补齐缺项。定位指标由定位infer产出；生成metrics使用项目严格coverage/provenance检查。连续帧仍按64帧manifest clip组织，FVD窗口16、stride16、size224，不跨clip。paired图像指标size448，推理seed42。

query 不能用于学习率、训练长度、checkpoint或最终辅助候选的选择；所有适配配方在观察Unseen结果前冻结。原协议为每个适配模型回测完整Seen-10并报告各自retention，不把四个模型的retention误当成同一个模型的成绩。根据后续计算预算安排，**剩余 retention 推理／评测已暂停**；保留已有结果和入口，当前只补训练与 query，不将缺少 retention 记为已完成。

### 2026-11-06：批量推理与结果兼容

`scripts/run_exp37_unseen4.py infer/metrics/aggregate` 支持 `--test-loc-batch-size`（默认 **32**）与 `--test-gen-batch-size`（默认 **16**）。分别用于定位和离散／连续生成；`--task all` 仍按 family 的单任务／联合路线分发，不要求单任务模型运行另一任务。zero-shot、few-shot query 以及保留的 retention 入口使用同一规则；参数不写入训练配置、不改变 400 步适配或来源权重身份。

| 情况 | 默认批量结果路径（相对 `outputs_loc/benchmark_v2` 或 `outputs_eval/benchmark_v2`） |
|---|---|
| 每图 few-shot 定位 | `{family}_{map}/shot_{N}/seed_0/infer_bs32/` |
| 每图 few-shot 生成 | `{family}_{map}/shot_{N}/seed_0/infer_bs16/{discrete,continuous}/` |
| zero-shot | `{family}/zero_shot/seed_42/infer_bs32/` 或 `infer_bs16/{discrete,continuous}/` |
| 四图 few-shot 汇总 | `{family}/shot_{N}/seed_0/infer_bs{B}/map_models/` |

保留的 retention 路径在 `infer_bs{B}` 后追加 `seen_retention`，生成再追加 discrete／continuous。`infer`、`metrics`、`aggregate` 要传同一组批量值；汇总校验每张地图的快照、执行元数据与原覆盖率，不混用不同 batch 的四图结果。显式设为 `1` 时沿用原路径、原快照格式；**只补旧结果的指标或四图汇总，必须显式传相应参数为 `1`**，无需重跑已完整的旧推理。

生成每批真正并行执行编码、20 步 DiT、VAE；定位为 10 步 flow。尾批不丢弃，连续帧保持 manifest 顺序与 clip 身份。按旧单样本次序抽取初始噪声，执行清单记录批量、样本顺序哈希、耗时／吞吐及峰值显存。保留 seed=42、CFG、分辨率、数据划分与指标定义。现有生成入口在加载后注入 LoRA，当前 PEFT 下新增 Dropout 可保持 training 模式；本次保留该行为并在 B>1 元数据中记录实际 active Dropout，未默默改为全模型 eval。其 mask 分配会随 batch 改变，故新旧结果的差异不只有浮点误差，不能假定逐像素相同。应固定批量比较新结果；关闭生成 Dropout 属于需单独评估的协议修正。未自动进行 OOM 降档或逐图片断点续跑。首次可用 GPU 上应先核验小批结果和显存，再安排完整 query。

Seen／Unseen 的可执行示例统一见 [record.md](record.md#exp37-batched-inference-20261106)。目前仅通过 CPU 路由、批处理及 RNG 回归，未占用本机正在运行任务的 GPU 验证真实模型的默认批量或测速；未启动任何完整实验。

`aggregate` 对同family同shot的四个目标图结果执行 `map-models` 等地图宏平均，保留map→checkpoint映射。zero-shot使用一个父模型覆盖四图，采用普通 `maps` 聚合；两者统计身份不同。单任务缺失的任务列写N/A，未执行的有效任务写待执行，不写0。

## 资产、来源与中断处理

默认使用 minimal 资产报告 `data/csgo_benchmark_v2/minimal_dataset_report.json`。首次训练使用全量资产时传 `--asset-mode full --data-dir /path/to/preprocessed_data`；只替换物理资产后端，不重新采样。后续推理/评测读取本次保存的resolved配置，不能把minimal/full的输出混用。zero-shot首次infer也须明确选择需要的后端。

首次适配执行前核对来源 `checkpoint-6000/model.safetensors`、`trainer_state.json`、来源根目录 `resolved_config.yaml` 和 `experiment_plan.json`；保存来源身份及权重指纹。缺少来源或实际step不是6000时拒绝执行，不寻找替代checkpoint。配置规划和CPU测试不需要读取真实大模型权重。2026-11-07 本机已同步 `exp37_s2_gen`、`exp37_s2_loc`、`exp37_s2`、`exp37_s2_both` 的上述四项来源文件，Trainer `global_step=max_steps=6000`；新到的 `exp37_s2_both` 具备 `exp37_fs_both` 所需来源文件，Seen评测同步进度见[阶段性结果](CSGO_BENCHMARK_V2_EXP37_SEEN10.md#exp37-s2-both-update-20261107)。`exp37_s2_both_b1` 仍缺这些来源文件。来源齐全不等于zero-shot或few-shot已运行；2026-11-07 both 已同步进度见[最新结果](#exp37-both-results-20261107)，本段保留来源文件核查时点。

本协议没有中途完整optimizer checkpoint，**不支持将中断的400步训练用 `--resume` 接续**。若训练中断，需要人工保留/归档本次不完整目录后，从同一个Seen来源重新运行该适配点。已完成400步而推理/评测中断时，只补对应infer/metrics，不重复train；训练入口拒绝覆盖已有运行。不要假定图像推理具有逐样本断点续跑功能。

<a id="exp37-both-progress-20261109"></a>

## 实验记录表

2026-11-09核对：下表每个shot单元格依次记录**训练终点本机核验／目标地图query／Seen-retention**。`✅`表示已同步完整证据；gen的query完成同时要求离散和连续两项齐全。32个单任务与16个both适配点的正式模型、Trainer状态均已在本机核验final400/400；对应三项任务的query单图指标共96份齐全（各任务32份）。both 的 `de_vertigo` 20-shot `seed_0.save_failed_20261107_171318` 是失败归档，已排除。retention部分产物与完整指标分开记录，剩余不补跑。

| Family | 目标地图 | 100-shot | 50-shot | 20-shot | 10-shot | 正式checkpoint |
|---|---|---|---|---|---|---|
| exp37_fs_gen | cs_office | ✅／✅／部分，暂停 | ✅／✅／部分，暂停 | ✅／✅／部分，暂停 | ✅／✅／部分，暂停 | local400 |
| exp37_fs_gen | de_golden | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | local400 |
| exp37_fs_gen | de_palacio | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | local400 |
| exp37_fs_gen | de_vertigo | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | local400 |
| exp37_fs_loc | cs_office | ✅／✅／✅ | ✅／✅／✅ | ✅／✅／✅ | ✅／✅／✅ | local400 |
| exp37_fs_loc | de_golden | ✅／✅／✅ | ✅／✅／✅ | ✅／✅／✅ | ✅／✅／✅ | local400 |
| exp37_fs_loc | de_palacio | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | local400 |
| exp37_fs_loc | de_vertigo | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | ✅／✅／暂停 | local400 |
| exp37_fs_joint | cs_office | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_joint | de_golden | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_joint | de_palacio | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_joint | de_vertigo | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both | cs_office | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both | de_golden | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both | de_palacio | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both | de_vertigo | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both_b1 | cs_office | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both_b1 | de_golden | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both_b1 | de_palacio | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both_b1 | de_vertigo | 未同步 | 未同步 | 未同步 | 未同步 | local400 |

zero-shot五组以及四图宏平均结果同步维护在 [`csgo_benchmark_v2_experiments_results.md`](csgo_benchmark_v2_experiments_results.md#exp37-unseen4-results-20261103)。both四档共16个训练点均已核验final400/400，48份单图query指标已齐。

## 已同步 zero-shot 结果（2026-11-07 补充）

下表直接采用原始 JSON 的 `metrics_macro_map`，即四张 Unseen 地图等权平均。定位的 XY／Z／Pitch／Yaw 均为距离误差（越低越好）；生成的 PSNR／SSIM／CLIP 越高越好，其余列越低越好。N/A 表示该来源不执行此任务。四个已同步来源均使用未适配的 Seen Stage2@6000 权重、推理 seed=42、minimal 资产后端；zero-shot 没有 support seed 或 shot。query 为 `crossmap_query_test`，每图2,000条；continuous 为 `crossmap_continuous`，每图20条×64帧。每项已有四图独立指标、推理清单、评测权重计划与聚合 JSON；生成每图 `Common_Count` 分别为2,000／1,280，continuous 每图 `Track_Count=20`。

| 来源／zero-shot | 定位 XY↓ | Z↓ | Pitch↓ | Yaw↓ | 离散 PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | CLIP↑ | 连续 PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp37_fs_gen` | N/A | N/A | N/A | N/A | 11.379 | 0.380 | 0.717 | 77.496 | 0.746 | 11.355 | 0.374 | 0.717 | 1283.955 |
| `exp37_fs_loc` | 339.928 | 19.874 | 7.610 | 91.572 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `exp37_fs_joint` | 344.490 | 18.720 | 7.816 | 92.099 | 11.041 | 0.374 | 0.729 | 103.583 | 0.733 | 11.106 | 0.369 | 0.728 | 1368.558 |
| `exp37_fs_both` | 341.5329 | 18.9907 | 9.0030 | 93.0790 | 11.0518 | 0.3706 | 0.7285 | 100.0587 | 0.7324 | 11.1011 | 0.3657 | 0.7289 | 1334.9445 |
| `exp37_fs_both_b1` | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |

both zero-shot 原始指标：[定位](outputs_loc/benchmark_v2/exp37_fs_both/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json)、[离散](outputs_eval/benchmark_v2/exp37_fs_both/zero_shot/seed_42/infer_bs32/discrete/summary.json)、[连续](outputs_eval/benchmark_v2/exp37_fs_both/zero_shot/seed_42/infer_bs32/continuous/summary.json)。定位 `infer_bs16`、两类生成 `infer_bs32`；旧 gen／loc／joint 三组仍为 batch=1。

原始聚合指标：[gen 离散](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/discrete/summary.json)、[gen 连续](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/continuous/summary.json)、[loc 定位](outputs_loc/benchmark_v2/exp37_fs_loc/zero_shot/seed_42/benchmark_csgo_v2_loc.json)、[joint 定位](outputs_loc/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/benchmark_csgo_v2_loc.json)、[joint 离散](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/discrete/summary.json)、[joint 连续](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/continuous/summary.json)。旧六项的 split／map／数量／seed／checkpoint 与各自 `inference_manifest.json`、`evaluation_plan.json` 一致；生成四图原始指标与清单一致。上述 zero-shot 数值可用于描述未适配模型的跨图表现，不能用于挑选适配配方或推断 few-shot 收益。

<a id="exp37-fewshot-results-20261106"></a>

## 已同步 few-shot 结果（2026-11-06）

本次只整理已有产物，不运行训练、推理或重新打分。已核对 **32个训练点、48个query指标文件与各自推理清单**：gen／loc分别继承 `exp37_s2_gen@6000`／`exp37_s2_loc@6000`；所有训练终点400/400，support seed=0、推理seed=42、minimal资产，地图、shot、checkpoint与计划一致。定位与离散生成每点各2000样本，连续生成每点1280帧、20个64帧clip；生成缺失／多余预测均为0，coverage=1。四档query总计定位32000条、离散生成32000张、连续生成20480帧。`de_palacio` 100-shot定位的旧中断归档目录已排除，只读取正式 `seed_0/`。

**这些完成结果均使用batch=1**，与对应family的zero-shot一致；不是新增默认定位32／生成16的结果。每张地图使用自己的适配模型，下面是四图等权宏平均，FID／FVD也是四个逐图指标的平均，不是把所有图片／视频混池重算。正式四图`map-models`聚合JSON尚未同步，表中macro从四个单图JSON直接复算；下方逐图表提供全部源链接。

### 定位：exp37_fs_loc（四图宏平均）

| Shot | XY↓ | Z↓ | Pitch↓ | Yaw↓ |
|---|---|---|---|---|
| 0-shot | 339.9282 | 19.8735 | 7.6095 | 91.5718 |
| 100-shot | 187.4802 | 13.2434 | 4.4404 | 76.9022 |
| 50-shot | 201.3729 | 14.2518 | 4.7805 | 82.1231 |
| 20-shot | 207.9682 | 14.8062 | 4.8209 | 84.2050 |
| 10-shot | 227.0811 | 15.0967 | 5.7064 | 86.3811 |

### 离散生成：exp37_fs_gen（四图宏平均）

| Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ |
|---|---|---|---|---|---|---|
| 0-shot | 11.3790 | 0.3805 | 0.7169 | 0.4578 | 77.4963 | 0.7460 |
| 100-shot | 13.2371 | 0.4347 | 0.6518 | 0.4857 | 37.0743 | 0.8293 |
| 50-shot | 12.9100 | 0.4273 | 0.6647 | 0.4793 | 39.4414 | 0.8213 |
| 20-shot | 12.6999 | 0.4262 | 0.6747 | 0.4704 | 43.3275 | 0.8175 |
| 10-shot | 12.6078 | 0.4214 | 0.6804 | 0.4624 | 48.3858 | 0.8085 |

### 连续生成：exp37_fs_gen（四图宏平均）

| Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ |
|---|---|---|---|---|---|---|---|---|
| 0-shot | 11.3554 | 0.3739 | 0.7167 | 0.4581 | 44.5674 | 50.1917 | 16.8619 | 1283.9551 |
| 100-shot | 13.1245 | 0.4203 | 0.6554 | 0.4875 | 37.9226 | 43.9832 | 16.6516 | 917.9839 |
| 50-shot | 12.5932 | 0.4089 | 0.6758 | 0.4734 | 38.5864 | 44.6535 | 16.5679 | 952.1893 |
| 20-shot | 12.3751 | 0.4086 | 0.6796 | 0.4684 | 36.6470 | 42.9859 | 16.7518 | 890.7293 |
| 10-shot | 12.2757 | 0.4018 | 0.6863 | 0.4627 | 36.5785 | 42.9157 | 17.0694 | 935.1674 |

T-Warp、T-Diff、OFE分别对应 `Temporal_Warping_Error`、`Temporal_Difference_Error`、`Optical_Flow_EPE`。连续生成仍按当前帧pose独立生成，评测沿用64帧clip、FVD窗口16／stride16／size224；生成指标使用 `benchmark_v2_core`，被该profile跳过的指标不填0。

### 逐图定位结果

| 目标地图 | Shot | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 原始指标 |
|---|---|---|---|---|---|---|
| cs_office | 100 | 111.6483 | 2.8293 | 4.3639 | 74.3903 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_100/seed_0/benchmark_csgo_v2_loc.json) |
| cs_office | 50 | 135.7396 | 3.3065 | 4.5929 | 83.6845 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_50/seed_0/benchmark_csgo_v2_loc.json) |
| cs_office | 20 | 151.4746 | 3.4999 | 4.8987 | 81.7117 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_20/seed_0/benchmark_csgo_v2_loc.json) |
| cs_office | 10 | 168.7539 | 3.7127 | 5.4874 | 84.7514 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_10/seed_0/benchmark_csgo_v2_loc.json) |
| de_golden | 100 | 195.8251 | 13.9793 | 3.5995 | 79.2260 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_100/seed_0/benchmark_csgo_v2_loc.json) |
| de_golden | 50 | 200.1612 | 14.2368 | 4.3510 | 84.7906 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_50/seed_0/benchmark_csgo_v2_loc.json) |
| de_golden | 20 | 220.6234 | 15.1666 | 3.9455 | 87.6958 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_20/seed_0/benchmark_csgo_v2_loc.json) |
| de_golden | 10 | 249.3597 | 15.7151 | 4.2349 | 86.6218 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_10/seed_0/benchmark_csgo_v2_loc.json) |
| de_palacio | 100 | 254.9931 | 16.6682 | 3.6813 | 71.5318 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_palacio/shot_100/seed_0/benchmark_csgo_v2_loc.json) |
| de_palacio | 50 | 272.9916 | 18.6016 | 3.5189 | 78.2519 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_palacio/shot_50/seed_0/benchmark_csgo_v2_loc.json) |
| de_palacio | 20 | 263.0241 | 18.7307 | 3.2822 | 81.5502 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_palacio/shot_20/seed_0/benchmark_csgo_v2_loc.json) |
| de_palacio | 10 | 283.5555 | 18.2876 | 3.5568 | 87.0781 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_palacio/shot_10/seed_0/benchmark_csgo_v2_loc.json) |
| de_vertigo | 100 | 187.4544 | 19.4967 | 6.1168 | 82.4606 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_vertigo/shot_100/seed_0/benchmark_csgo_v2_loc.json) |
| de_vertigo | 50 | 196.5993 | 20.8622 | 6.6594 | 81.7654 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_vertigo/shot_50/seed_0/benchmark_csgo_v2_loc.json) |
| de_vertigo | 20 | 196.7506 | 21.8275 | 7.1572 | 85.8624 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_vertigo/shot_20/seed_0/benchmark_csgo_v2_loc.json) |
| de_vertigo | 10 | 206.6552 | 22.6714 | 9.5465 | 87.0732 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_vertigo/shot_10/seed_0/benchmark_csgo_v2_loc.json) |

### 逐图离散生成结果

| 目标地图 | Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | 原始指标 |
|---|---|---|---|---|---|---|---|---|
| cs_office | 100 | 13.6081 | 0.5756 | 0.6285 | 0.4585 | 36.1762 | 0.8569 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_100/seed_0/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 50 | 13.2860 | 0.5711 | 0.6539 | 0.4448 | 38.7350 | 0.8472 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_50/seed_0/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 20 | 12.6904 | 0.5685 | 0.6715 | 0.4343 | 44.9956 | 0.8389 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_20/seed_0/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 10 | 12.6166 | 0.5672 | 0.6795 | 0.4189 | 50.2244 | 0.8201 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_10/seed_0/discrete/benchmark_csgo_v2_cs_office.json) |
| de_golden | 100 | 14.7201 | 0.3692 | 0.6190 | 0.5216 | 38.3630 | 0.8226 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_100/seed_0/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 50 | 14.6109 | 0.3641 | 0.6272 | 0.5162 | 37.5379 | 0.8128 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_50/seed_0/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 20 | 14.3799 | 0.3614 | 0.6343 | 0.5083 | 39.3602 | 0.8141 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_20/seed_0/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 10 | 14.2665 | 0.3440 | 0.6349 | 0.5021 | 48.0172 | 0.8183 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_10/seed_0/discrete/benchmark_csgo_v2_de_golden.json) |
| de_palacio | 100 | 12.6608 | 0.3556 | 0.6962 | 0.5087 | 36.5895 | 0.8164 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_100/seed_0/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 50 | 12.1019 | 0.3439 | 0.7041 | 0.5067 | 41.4511 | 0.8133 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_50/seed_0/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 20 | 12.2056 | 0.3387 | 0.7106 | 0.4980 | 42.8269 | 0.8136 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_20/seed_0/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 10 | 12.0682 | 0.3196 | 0.7150 | 0.4879 | 47.1807 | 0.8007 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_10/seed_0/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_vertigo | 100 | 11.9593 | 0.4383 | 0.6634 | 0.4538 | 37.1684 | 0.8215 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_100/seed_0/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 50 | 11.6411 | 0.4300 | 0.6737 | 0.4495 | 40.0417 | 0.8119 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_50/seed_0/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 20 | 11.5236 | 0.4364 | 0.6825 | 0.4408 | 46.1273 | 0.8036 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_20/seed_0/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 10 | 11.4799 | 0.4547 | 0.6921 | 0.4407 | 48.1207 | 0.7950 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_10/seed_0/discrete/benchmark_csgo_v2_de_vertigo.json) |

### 逐图连续生成结果

| 目标地图 | Shot | PSNR↑ | SSIM↑ | LPIPS↓ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 原始指标 |
|---|---|---|---|---|---|---|---|---|---|
| cs_office | 100 | 13.6717 | 0.5467 | 0.6424 | 34.7387 | 40.8071 | 14.3160 | 713.1726 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_100/seed_0/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 50 | 12.7497 | 0.5321 | 0.6856 | 36.2184 | 42.2901 | 14.2711 | 653.0254 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_50/seed_0/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 20 | 12.3197 | 0.5316 | 0.6930 | 36.2595 | 42.4213 | 15.3887 | 697.2700 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_20/seed_0/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 10 | 12.1304 | 0.5254 | 0.7081 | 37.8227 | 44.0237 | 16.1816 | 781.7953 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_10/seed_0/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| de_golden | 100 | 14.7398 | 0.3652 | 0.6159 | 31.7486 | 36.9833 | 17.7700 | 1125.3607 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_100/seed_0/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 50 | 14.6808 | 0.3602 | 0.6281 | 30.5593 | 36.0256 | 17.6124 | 1177.3837 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_50/seed_0/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 20 | 14.3139 | 0.3542 | 0.6315 | 29.0418 | 34.7189 | 17.4883 | 1021.4769 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_20/seed_0/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 10 | 14.1996 | 0.3391 | 0.6295 | 28.6820 | 34.4263 | 17.8508 | 951.9031 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_10/seed_0/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_palacio | 100 | 12.4010 | 0.3386 | 0.7007 | 40.8908 | 47.2262 | 17.8149 | 1004.6232 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_100/seed_0/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 50 | 11.7388 | 0.3297 | 0.7162 | 43.9713 | 50.0024 | 18.1769 | 1067.5798 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_50/seed_0/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 20 | 11.9449 | 0.3309 | 0.7075 | 39.4967 | 45.9608 | 18.2695 | 964.8729 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_20/seed_0/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 10 | 11.6472 | 0.3071 | 0.7104 | 40.7659 | 47.0170 | 19.0299 | 1044.6473 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_10/seed_0/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_vertigo | 100 | 11.6853 | 0.4305 | 0.6624 | 44.3123 | 50.9162 | 16.7055 | 828.7789 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_100/seed_0/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 50 | 11.2036 | 0.4136 | 0.6735 | 43.5965 | 50.2959 | 16.2111 | 910.7683 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_50/seed_0/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 20 | 10.9217 | 0.4179 | 0.6862 | 41.7901 | 48.8426 | 15.8606 | 879.2974 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_20/seed_0/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 10 | 11.1254 | 0.4354 | 0.6973 | 39.0433 | 46.1959 | 15.2152 | 962.3239 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_10/seed_0/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |

<a id="exp37-retention-observed-20261106"></a>

### 已完成的 retention：仅记录现有结果，剩余暂停

定位已有 `cs_office` 与 `de_golden` 各四种shot、共8组完整Seen-10回测；每行是**该目标图适配模型在整个Seen-10的宏平均**，不是目标Unseen地图query。其余两图的8组定位retention暂停。生成retention未完成任何完整指标：office的100／50／20／10-shot离散输出分别有9226／9539／9833／10131张图片（各应为20000），连续仅有配置／计划且无图片；均缺完成清单和指标，其他地图无完整retention产物。原有部分输出保留，不纳入宏平均，不自动补跑。

| 适配目标地图／来源 | Shot | Seen XY↓ | Z↓ | Pitch↓ | Yaw↓ | 原始指标 |
|---|---|---|---|---|---|---|
| 未适配来源 exp37_s2_loc | — | 46.3528 | 2.8725 | 2.5183 | 25.1111 | [JSON](outputs_loc/benchmark_v2/exp37_s2_loc/checkpoint-6000/test/seed_42/benchmark_csgo_v2_loc.json) |
| cs_office | 100 | 208.9502 | 14.4411 | 4.5273 | 36.4528 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_100/seed_0/seen_retention/benchmark_csgo_v2_loc.json) |
| cs_office | 50 | 225.2160 | 16.8247 | 3.1864 | 38.9405 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_50/seed_0/seen_retention/benchmark_csgo_v2_loc.json) |
| cs_office | 20 | 236.5602 | 13.8942 | 2.9682 | 50.6343 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_20/seed_0/seen_retention/benchmark_csgo_v2_loc.json) |
| cs_office | 10 | 172.5722 | 12.2462 | 3.4795 | 48.0637 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_10/seed_0/seen_retention/benchmark_csgo_v2_loc.json) |
| de_golden | 100 | 83.6171 | 4.8722 | 2.7911 | 35.1428 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_100/seed_0/seen_retention/benchmark_csgo_v2_loc.json) |
| de_golden | 50 | 73.9879 | 4.6084 | 2.9303 | 50.8554 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_50/seed_0/seen_retention/benchmark_csgo_v2_loc.json) |
| de_golden | 20 | 65.4182 | 3.6386 | 2.7422 | 39.3333 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_20/seed_0/seen_retention/benchmark_csgo_v2_loc.json) |
| de_golden | 10 | 76.0354 | 6.9483 | 2.8575 | 33.9378 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_10/seed_0/seen_retention/benchmark_csgo_v2_loc.json) |

已完成8组在相同Seen-test、batch=1、seed42口径下的四项定位误差均高于未适配来源。Seen XY由46.3528升至65.4182～236.5602（约1.41～5.10倍），说明这些已测模型存在适配后原Seen能力下降；不能写成“few-shot提升且没有遗忘”。该结论仅覆盖已测定位模型，不外推另外两图和生成retention，也不要求恢复已暂停的评测。

<a id="exp37-both-results-20261107"></a>
<a id="exp37-both-results-20261109"></a>

## 2026-11-09：exp37_fs_both（P4＋U2）完整结果与同步状态

四图100／50／20／10-shot共16个独立适配点的定位、离散、连续query指标现已全部同步，合计**16/16点、48/48份**。每点定位与离散各2000条／张、连续1280帧及20个64帧clip，coverage=1、缺失／多余预测为0。query结果无缺失；正式训练权重、Trainer状态及配置计划文件均已在本机核验，进度见[进度表](#exp37-both-progress-20261109)。

定位 `infer_bs16`、生成 `infer_bs32`，推理seed=42、support seed=0、minimal资产。0-shot来自 `exp37_s2_both@6000` 的原有四图聚合指标。100／50-shot的三项正式 `map-models` 汇总均已同步；50-shot来源：[定位](outputs_loc/benchmark_v2/exp37_fs_both/shot_50/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json)、[离散](outputs_eval/benchmark_v2/exp37_fs_both/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json)、[连续](outputs_eval/benchmark_v2/exp37_fs_both/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json)，100／50-shot汇总均与各自四个单图JSON等权复算一致。100-shot正式汇总：[定位](outputs_loc/benchmark_v2/exp37_fs_both/shot_100/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json)、[离散](outputs_eval/benchmark_v2/exp37_fs_both/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json)、[连续](outputs_eval/benchmark_v2/exp37_fs_both/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json)。10／20-shot四图macro由单图JSON等权复算，正式 `map-models` 汇总尚未同步。FID／FVD也按四图等权。`de_vertigo` 20-shot的 `seed_0.save_failed_20261107_171318` 是失败归档，仅正式 `seed_0/` 入表。

生成元数据包含438个 activeDropout 且 `stochastic_layers_batch_dependent=true`；both 的 loc16／gen32 与旧 gen／loc／joint 的 batch=1 不具备严格一致的推理口径，不能由跨family差异作辅助loss归因。原始few-shot指标保留远端绝对路径；本机按相同相对输出身份核对路径、manifest、资产摘要、batch和覆盖，未改JSON或重新运行metric。原样在本机运行严格校验命令需映射旧服务器绝对路径。

### 定位：四图等权 macro

| Shot | XY↓ | Z↓ | Pitch↓ | Yaw↓ |
|---:|---:|---:|---:|---:|
| 0-shot | 341.5329 | 18.9907 | 9.0030 | 93.0790 |
| 100-shot | 171.8577 | 11.9646 | 4.8400 | 74.9741 |
| 50-shot | 188.9630 | 13.2528 | 5.0187 | 80.8331 |
| 20-shot | 206.1598 | 14.3399 | 5.7314 | 85.0548 |
| 10-shot | 222.5440 | 15.1374 | 6.2334 | 88.4703 |

### 定位：逐图结果

| 目标地图 | Shot | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 原始指标 |
|---|---:|---:|---:|---:|---:|---|
| cs_office | 100 | 100.8698 | 2.3594 | 5.4618 | 72.6795 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_cs_office/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 50 | 130.2954 | 2.8049 | 6.0422 | 85.5553 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_cs_office/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 20 | 142.4106 | 3.5887 | 7.1826 | 86.9190 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_cs_office/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 10 | 151.4895 | 3.8976 | 6.1603 | 88.9383 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_cs_office/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 100 | 185.2977 | 13.3480 | 4.0590 | 80.1399 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_golden/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 50 | 191.6503 | 14.3819 | 4.3581 | 83.9499 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_golden/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 20 | 214.7585 | 16.2936 | 5.1563 | 87.2698 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_golden/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 10 | 234.4089 | 16.9440 | 5.5182 | 87.0724 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_golden/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 100 | 228.5884 | 14.5007 | 3.6786 | 67.7949 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_palacio/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 50 | 251.7280 | 16.1849 | 3.7664 | 71.5218 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_palacio/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 20 | 269.3269 | 15.4318 | 4.2127 | 77.9488 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_palacio/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 10 | 292.0618 | 16.4159 | 4.3698 | 85.4077 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_palacio/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 100 | 172.6749 | 17.6503 | 6.1607 | 79.2822 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_vertigo/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 50 | 182.1782 | 19.6395 | 5.9083 | 82.3053 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_vertigo/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 20 | 198.1434 | 22.0457 | 6.3740 | 88.0816 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_vertigo/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 10 | 212.2157 | 23.2922 | 8.8853 | 92.4629 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both_de_vertigo/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |

### 离散生成：四图等权 macro

| Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 0-shot | 11.0518 | 0.3706 | 0.7285 | 0.4604 | 100.0587 | 0.7324 |
| 100-shot | 13.0418 | 0.4293 | 0.6673 | 0.4807 | 39.5391 | 0.8154 |
| 50-shot | 12.8144 | 0.4265 | 0.6747 | 0.4761 | 40.3697 | 0.8124 |
| 20-shot | 12.5950 | 0.4255 | 0.6826 | 0.4687 | 45.3569 | 0.8053 |
| 10-shot | 12.4688 | 0.4152 | 0.6883 | 0.4641 | 51.5688 | 0.8022 |

### 离散生成：逐图结果

| 目标地图 | Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | 原始指标 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cs_office | 100 | 13.1996 | 0.5636 | 0.6524 | 0.4441 | 40.9455 | 0.8490 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_cs_office/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 50 | 13.0743 | 0.5745 | 0.6605 | 0.4382 | 42.0617 | 0.8398 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_cs_office/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 20 | 12.6453 | 0.5768 | 0.6715 | 0.4277 | 48.9047 | 0.8358 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_cs_office/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 10 | 12.7838 | 0.5498 | 0.6792 | 0.4189 | 53.6271 | 0.8246 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_cs_office/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| de_golden | 100 | 14.4418 | 0.3662 | 0.6325 | 0.5176 | 39.8082 | 0.8028 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_golden/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 50 | 14.3736 | 0.3651 | 0.6369 | 0.5163 | 38.6305 | 0.8039 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_golden/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 20 | 14.2202 | 0.3483 | 0.6436 | 0.5093 | 41.8457 | 0.7957 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_golden/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 10 | 13.8478 | 0.3371 | 0.6475 | 0.5046 | 48.8016 | 0.8082 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_golden/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_palacio | 100 | 12.5013 | 0.3472 | 0.7084 | 0.5101 | 39.0612 | 0.8005 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_palacio/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 50 | 12.3716 | 0.3420 | 0.7183 | 0.5031 | 39.7537 | 0.7991 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_palacio/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 20 | 12.1644 | 0.3473 | 0.7232 | 0.4997 | 44.2656 | 0.7931 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_palacio/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 10 | 11.8333 | 0.3305 | 0.7264 | 0.4904 | 51.7261 | 0.7898 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_palacio/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_vertigo | 100 | 12.0245 | 0.4403 | 0.6757 | 0.4510 | 38.3416 | 0.8094 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_vertigo/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 50 | 11.4382 | 0.4246 | 0.6832 | 0.4468 | 41.0331 | 0.8066 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_vertigo/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 20 | 11.3501 | 0.4295 | 0.6922 | 0.4380 | 46.4118 | 0.7969 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_vertigo/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 10 | 11.4104 | 0.4433 | 0.7000 | 0.4425 | 52.1205 | 0.7862 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_vertigo/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |

### 连续生成：四图等权 macro

| Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-shot | 11.1011 | 0.3657 | 0.7289 | 0.4581 | 43.0778 | 48.9258 | 16.9266 | 1334.9445 |
| 100-shot | 12.8341 | 0.4119 | 0.6730 | 0.4846 | 40.3321 | 46.2007 | 16.3582 | 984.8784 |
| 50-shot | 12.4543 | 0.4072 | 0.6850 | 0.4765 | 40.4484 | 46.3713 | 16.4332 | 1030.1365 |
| 20-shot | 12.3794 | 0.4088 | 0.6889 | 0.4725 | 38.7084 | 44.8006 | 16.5945 | 950.1136 |
| 10-shot | 12.2896 | 0.3971 | 0.6932 | 0.4662 | 37.4045 | 43.6475 | 16.7706 | 933.5771 |

### 连续生成：逐图结果

| 目标地图 | Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 原始指标 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cs_office | 100 | 12.9196 | 0.5287 | 0.6679 | 0.4507 | 39.2649 | 45.0055 | 14.5538 | 704.0578 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_cs_office/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 50 | 12.5058 | 0.5351 | 0.6892 | 0.4306 | 37.5142 | 43.7319 | 14.2201 | 751.1638 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_cs_office/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 20 | 12.1699 | 0.5378 | 0.6863 | 0.4436 | 35.9450 | 42.2949 | 14.5927 | 832.0142 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_cs_office/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 10 | 12.6481 | 0.5146 | 0.6985 | 0.4221 | 37.0186 | 43.1744 | 15.8381 | 832.8553 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_cs_office/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| de_golden | 100 | 14.4747 | 0.3600 | 0.6305 | 0.5258 | 34.1063 | 39.1925 | 17.5258 | 1153.4139 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_golden/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 50 | 14.2368 | 0.3578 | 0.6384 | 0.5217 | 34.6652 | 39.7623 | 17.5321 | 1217.6998 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_golden/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 20 | 14.3498 | 0.3444 | 0.6452 | 0.5141 | 32.5848 | 37.7831 | 18.0469 | 1025.2139 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_golden/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 10 | 13.8450 | 0.3303 | 0.6371 | 0.5153 | 33.1870 | 38.3790 | 18.1840 | 960.8682 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_golden/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_palacio | 100 | 12.2570 | 0.3306 | 0.7197 | 0.5110 | 41.5991 | 47.8714 | 17.3676 | 1199.8149 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_palacio/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 50 | 11.9723 | 0.3221 | 0.7306 | 0.5074 | 42.4912 | 48.4397 | 17.6372 | 1067.8219 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_palacio/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 20 | 12.0625 | 0.3403 | 0.7265 | 0.5007 | 41.4207 | 47.5213 | 17.6013 | 982.6016 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_palacio/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 10 | 11.9303 | 0.3263 | 0.7304 | 0.4931 | 40.4969 | 46.8213 | 17.7553 | 1003.1593 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_palacio/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_vertigo | 100 | 11.6851 | 0.4282 | 0.6738 | 0.4508 | 46.3582 | 52.7334 | 15.9858 | 882.2270 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_vertigo/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 50 | 11.1021 | 0.4137 | 0.6820 | 0.4464 | 47.1229 | 53.5514 | 16.3434 | 1083.8604 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_vertigo/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 20 | 10.9353 | 0.4126 | 0.6976 | 0.4317 | 44.8832 | 51.6030 | 16.1372 | 960.6246 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_vertigo/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 10 | 10.7349 | 0.4170 | 0.7070 | 0.4343 | 38.9155 | 46.2153 | 15.3051 | 937.4257 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both_de_vertigo/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |

### 本机同步状态与剩余项目

- **query结果无缺失**：16/16适配点、48/48份单图指标；100／50-shot正式 `map-models` 三项各齐。10／20-shot正式汇总JSON尚未同步，本页macro直接由四图单图JSON等权复算。
- 正式训练权重／Trainer状态已全部同步并核验：16/16个both点final400/400，和32个单任务点合计48个训练终点；query指标96份齐全，完成性分别见[进度表](#exp37-both-progress-20261109)。`exp37_fs_joint` 与 `exp37_fs_both_b1` 是独立family，没有本机few-shot完成证据。Seen-retention按用户决定暂停，未有both retention证据。

<a id="exp37-both-analysis-20261109"></a>

## 2026-11-09：both few-shot结果分析

下表均为四张Unseen地图等权宏平均；0-shot使用未适配的Seen Stage2@6000模型，10／20／50／100-shot各由四个独立目标图模型构成。定位、离散和连续分别使用上述原始JSON与推理口径。

| Shot | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 离散PSNR↑ | 离散LPIPS↓ | FID↓ | 连续PSNR↑ | 连续LPIPS↓ | FVD↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 341.5329 | 18.9907 | 9.0030 | 93.0790 | 11.0518 | 0.7285 | 100.0587 | 11.1011 | 0.7289 | 1334.9445 |
| 10 | 222.5440 | 15.1374 | 6.2334 | 88.4703 | 12.4688 | 0.6883 | 51.5688 | 12.2896 | 0.6932 | 933.5771 |
| 20 | 206.1598 | 14.3399 | 5.7314 | 85.0548 | 12.5950 | 0.6826 | 45.3569 | 12.3794 | 0.6889 | 950.1136 |
| 50 | 188.9630 | 13.2528 | 5.0187 | 80.8331 | 12.8144 | 0.6747 | 40.3697 | 12.4543 | 0.6850 | 1030.1365 |
| 100 | 171.8577 | 11.9646 | 4.8400 | 74.9741 | 13.0418 | 0.6673 | 39.5391 | 12.8341 | 0.6730 | 984.8784 |

1. 按既定协议，100-shot相对zero-shot的XY误差降低49.68%、FID降低60.48%、FVD降低26.22%。定位和生成均显示目标图适配收益；这只是观察结果，不能据此证明辅助损失的因果贡献。
2. 10→20→50→100-shot的四项定位宏误差逐步降低，离散PSNR、LPIPS、FID也逐步改善。各点均训练400次optimizer update，但joint base样本曝光分别为3200／8000／19200／40000，不能称等FLOPs，也不能把变化只归因于独立support样本数。
3. 连续PSNR、LPIPS逐步改善，FVD依次为933.5771／950.1136／1030.1365／984.8784，非单调且100-shot高于10-shot。不能根据Unseen query重新选择shot或其它超参数，也不能据此断言过拟合。
4. 与旧单任务100-shot作描述性比较：both XY 171.8577 vs loc 187.4802（−8.33%），Z −9.66%、Yaw −2.51%、Pitch +9.00%；both离散PSNR 13.0418 vs gen 13.2371（−0.1953 dB）、FID 39.5391 vs 37.0743（+6.65%）；连续FVD 984.8784 vs 917.9839（+7.29%）。这些观测值未显示双任务全面超过单任务。both定位／生成推理batch=16／32，旧单任务为1，且生成含438个activeDropout、`stochastic_layers_batch_dependent=true`；Seen初始化及历史训练预算也不同。因此这些跨family差值不能归因于辅助loss。
5. 同协议 `exp37_fs_joint` few-shot结果仍缺，不能检验“辅助 > joint > single”。Seen-retention已暂停，暂无both retention证据，不能宣称保留Seen能力。下一步应冻结配方、补齐并同步joint对照和匹配推理口径；不使用Unseen query选择方法。本次仅整理已有结果，未运行实验。

## 2026-11-06：单任务 few-shot 实验分析与总结（历史快照；[最新 both 同步结果](#exp37-both-results-20261107)）

1. **10-shot已经有明显目标图适配收益。** 相对同family zero-shot，定位XY由339.9282降至227.0811（下降33.20%）；离散FID由77.4963降至48.3858（下降37.56%）；连续FVD由1283.9551降至935.1674（下降27.16%）。四张图的这三项指标分别均好于自身zero-shot，收益不只来自一张地图。
2. **增加shot继续改善定位和逐帧生成，但不能称为等计算预算。** 按10→20→50→100，四图macro的四项定位误差、离散主要图像指标以及连续PSNR／SSIM／LPIPS均依次改善。100-shot较zero-shot的XY下降44.85%、FID下降52.16%；较10-shot的XY再下降17.44%、FID再下降23.38%、离散PSNR增加0.629 dB。各组虽然同为400次更新，但训练batch=shots，曝光从4000增至40000；这是既定few-shot协议下的整体结果，不能拆解为单纯增加独立样本的因果收益，也不能称等FLOPs。
3. **地图和定位维度仍有明显差异。** office的100-shot XY改善74.57%，贡献四图XY绝对降幅约53.68%；golden／palacio／vertigo的相对改善约30%，macro的44.85%不能代表每图。palacio适配后XY仍最高（254.9931），其20→50-shot曾从263.0241反升至272.9916。Yaw的100-shot宏误差仍为76.9022，比zero-shot仅下降16.02%，相对位置误差改善较弱。不能把总体趋势表述为“所有地图、所有指标严格单调”。
4. **连续逐帧质量改善，不等于时序指标随shot改善。** FVD按10／20／50／100-shot分别为935.1674／890.7293／952.1893／917.9839；100-shot相对10-shot只下降1.84%。各图最低FVD分别出现在office 50-shot、golden 10-shot、palacio 20-shot、vertigo 100-shot，T-Warp／T-Diff也不单调。20-shot当前宏FVD最低是报告事实，不能据Unseen query选择shot、checkpoint或重新调配方，更不能仅凭该曲线诊断为过拟合。
5. **已有局部retention证据显示适配代价。** office目标图的XY收益最大，同时已测Seen XY退化比golden更明显；8组已测retention的XY均高于来源。完整retention仍缺失，保持暂停安排；报告目标图收益时一并说明这项已观察到的取舍，不宣称保留全部Seen能力。
6. **当前建立的是两个单任务适配基线。** 这两组没有启用辅助loss，截至2026-11-06尚无匹配的joint／both／both_b1 few-shot结果，不能据此证明联合训练或辅助loss更优。下一优先按已经冻结的协议补齐joint的四图四shot对照，再评估预先由Seen validation确定的辅助候选；不使用本轮Unseen query来重新挑选辅助配方。连续生成分析保留逐帧和时序两组指标，可基于已生成固定clip检查差异；本次不启动新实验，也不恢复剩余retention。

总结果表同步见 [exp37 Unseen-4 结果专区](csgo_benchmark_v2_experiments_results.md#exp37-unseen4-results-20261103)。

## 历史工作与执行优先级

当前gen／loc／both共48个适配点的final400训练终点和96份query单图指标均已核验；100／50-shot正式汇总齐，10／20-shot正式汇总未同步但四图macro已由单图复算。joint few-shot仍缺本机证据；详见[2026-11-09结果](#exp37-both-results-20261109)与[分析](#exp37-both-analysis-20261109)。下一步冻结配方补齐同协议joint对照并匹配推理口径；不依据Unseen query选择方法。剩余retention继续暂停。

原主线若需补齐：`exp35` joint缺四图×四shot=16次，`exp35_gen/loc`各缺四图×20/10-shot=8次，共32次历史适配；`exp35_gen_de_vertigo` 50-shot本地已有四类推理manifest但缺metric摘要；`exp31_loc/exp32_loc`原表尚缺Unseen zero-shot。`exp36_1`四图四shot已登记完成；这些结果保持原来源和配方，不改名充作exp37结果。历史配置和手动命令已存在，本次不重新创建或启动它们。

## 原实现阶段验收边界（2026-11-03）

2026-11-03当时只执行配置校验、命令计划、CPU单元测试和shell语法检查，未运行train/eval/benchmark子进程，未加载真实模型。该历史实现快照不代表当前实验状态；2026-11-06已同步的32个单任务400步适配及完整query结果见本页新结果节。配置存在本身仍不等于实验完成。


## 实现验收（2026-11-03）

- `python -B -m unittest tests.test_run_exp37_unseen4 tests.test_fewshot_sampler_epoch -v`：15项CPU测试通过。包含真实sampler的索引检查，以及临时小文件模拟的来源6000／适配400、full后端继承、定位摘要、生成provenance和错误权重／shot／来源拒绝；未加载真实模型、未运行实验子进程。
- `python -B scripts/run_exp37_unseen4.py validate`：80个适配点通过；minimal/full两后端的870个train/infer/metrics/aggregate及zero-shot只读计划检查通过。
- `record.md` 新增六段shell代码均通过 `bash -n`，保留CRLF；新记录表20个family-map行、60个任务macro占位齐全，无伪造指标。新增文档链接、固定矩阵字段和脚本参数一致性已核查。
- `git diff --check`通过；原exp35/36/37 YAML以及旧3maps/map-specific runner未改。训练入口的epoch转发新增开关默认关闭，仅新few-shot开启。
- 原Seen37回归发现工作区另有Stage2 save_steps=1000与旧测试期望2000的差异；本次未修改该并行变更，不将其记为本次few-shot配方变更。
- 2026-11-03 当时三个输出根目录下未创建任何 `exp37_fs*` 实验目录，本机缺少五个Seen Stage2完整来源。这是实现阶段的历史快照；2026-11-04的zero-shot及2026-11-06的few-shot同步状态以上节为准。
