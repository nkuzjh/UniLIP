# CSGO Benchmark v2：exp37 Unseen-4 每图独立 few-shot

状态（2026-11-13 本机同步核查）：gen／loc／joint／both按定位bs16、生成bs32完成全部Unseen-4主线：64/64个final400模型、9/9项四图zero-shot、144/144份few-shot逐图指标和36/36份正式四图汇总均齐全。最新数值见[对齐batch完整结果](#exp37-unseen4-aligned-results-20261113)、[分析](#exp37-unseen4-aligned-analysis-20261113)；历史bs1结果保留。both_b1排除在本轮范围外，Seen-retention继续暂停。

历史状态（2026-11-12 本机同步核查）：`exp37_fs_joint` 四图×100／50／20／10-shot共16个适配模型均已完成final400；定位batch16、离散／连续生成batch32的48份逐图query指标及12份正式四图 `map-models` 汇总已齐全，详见[完整结果](#exp37-joint-results-20261112)与[比较分析](#exp37-joint-analysis-20261112)。joint／gen目前的zero-shot仍为历史batch1，不能直接与本轮few-shot计算同batch增益；Seen-retention继续暂停。

历史状态（2026-11-11 本机同步核查）：`exp37_fs_gen` 原有16个final400适配模型的生成query现另有 batch32 完整结果：四图×四档shot×离散／连续共32/32项推理、32/32份逐图指标和8/8份正式 `map-models` 汇总均齐全；详见[本轮结果](#exp37-gen-bs32-results-20261111)。这是同一批已训练模型的重新推理／评测，未增加训练点；旧batch1结果保留，gen batch32 zero-shot仍未同步，retention继续暂停。

历史状态（2026-11-09 本机同步核查）：**`exp37_fs_gen`／`exp37_fs_loc` 各16组与 `exp37_fs_both` 四图四档16组，共48个final400训练终点已独立核验**。both 四图四档共16/16点的定位、离散、连续query指标全部齐全（48/48份）；正式训练权重与Trainer状态均已同步并核验400/400。both 定位 infer_bs16、生成 infer_bs32；zero-shot三项齐全。旧 gen／loc／joint 为 batch=1；跨batch结果不可严格归因辅助loss。joint／both_b1 few-shot 仍无本机完成证据；retention暂停。协议、运行入口和手动命令见后文。

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

2026-11-13更新：当前主线gen／loc／joint／both共64个final400终点、9项四图zero-shot、144份few-shot逐图指标与36份正式汇总均已核对齐全。每个shot单元格依次记录**训练终点／目标地图query／Seen-retention**。当前定位bs16、生成bs32；gen／loc旧bs1 query另行保留，同一组训练权重不因推理batch变化重复计数。retention已有结果仍为原bs1口径，剩余暂停；both_b1为本轮范围外。失败归档 `seed_0.save_failed_*` 不计入正式结果。

| Family | 目标地图 | 100-shot | 50-shot | 20-shot | 10-shot | 正式checkpoint |
|---|---|---|---|---|---|---|
| exp37_fs_gen | cs_office | ✅／✅（bs1＋bs32）／部分，暂停 | ✅／✅（bs1＋bs32）／部分，暂停 | ✅／✅（bs1＋bs32）／部分，暂停 | ✅／✅（bs1＋bs32）／部分，暂停 | local400 |
| exp37_fs_gen | de_golden | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | local400 |
| exp37_fs_gen | de_palacio | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | local400 |
| exp37_fs_gen | de_vertigo | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | ✅／✅（bs1＋bs32）／暂停 | local400 |
| exp37_fs_loc | cs_office | ✅400/400／✅定位（bs1＋bs16）／✅（bs1历史） | ✅400/400／✅定位（bs1＋bs16）／✅（bs1历史） | ✅400/400／✅定位（bs1＋bs16）／✅（bs1历史） | ✅400/400／✅定位（bs1＋bs16）／✅（bs1历史） | local400 |
| exp37_fs_loc | de_golden | ✅400/400／✅定位（bs1＋bs16）／✅（bs1历史） | ✅400/400／✅定位（bs1＋bs16）／✅（bs1历史） | ✅400/400／✅定位（bs1＋bs16）／✅（bs1历史） | ✅400/400／✅定位（bs1＋bs16）／✅（bs1历史） | local400 |
| exp37_fs_loc | de_palacio | ✅400/400／✅定位（bs1＋bs16）／暂停 | ✅400/400／✅定位（bs1＋bs16）／暂停 | ✅400/400／✅定位（bs1＋bs16）／暂停 | ✅400/400／✅定位（bs1＋bs16）／暂停 | local400 |
| exp37_fs_loc | de_vertigo | ✅400/400／✅定位（bs1＋bs16）／暂停 | ✅400/400／✅定位（bs1＋bs16）／暂停 | ✅400/400／✅定位（bs1＋bs16）／暂停 | ✅400/400／✅定位（bs1＋bs16）／暂停 | local400 |
| exp37_fs_joint | cs_office | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_joint | de_golden | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_joint | de_palacio | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_joint | de_vertigo | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both | cs_office | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both | de_golden | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both | de_palacio | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both | de_vertigo | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | ✅400/400／✅定位+离散+连续／暂停 | final400已核验 |
| exp37_fs_both_b1 | cs_office | 本轮排除 | 本轮排除 | 本轮排除 | 本轮排除 | local400 |
| exp37_fs_both_b1 | de_golden | 本轮排除 | 本轮排除 | 本轮排除 | 本轮排除 | local400 |
| exp37_fs_both_b1 | de_palacio | 本轮排除 | 本轮排除 | 本轮排除 | 本轮排除 | local400 |
| exp37_fs_both_b1 | de_vertigo | 本轮排除 | 本轮排除 | 本轮排除 | 本轮排除 | local400 |

四个选定family的当前zero-shot和few-shot正式宏平均同步维护在[总结果表](csgo_benchmark_v2_experiments_results.md#exp37-unseen4-results-20261103)。全部定位bs16、生成bs32结果见[本轮完整表](#exp37-unseen4-aligned-results-20261113)；旧bs1保留在历史章节，both_b1规划保留但不纳入完成性统计。

<a id="exp37-unseen4-aligned-results-20261113"></a>

## 2026-11-13：对齐 batch 的完整 Unseen-4 结果

本轮主线只包含 `exp37_fs_gen`、`exp37_fs_loc`、`exp37_fs_joint`、`exp37_fs_both`；`exp37_fs_both_b1` 保留原规划记录，但排除在本轮完成性统计和结果比较之外。统一定位 **bs16**、离散／连续生成 **bs32**，不使用入口默认定位32／生成16替代实际执行批量。原bs1表格与历史分析继续保留，以下新结果不覆盖旧值。

| Family | final400训练终点 | zero-shot任务 | 四图四档few-shot任务／指标 | 正式四图few-shot汇总 |
|---|---|---|---|---|
| exp37_fs_gen | 16/16 | 2/2 | 32/32 | 8/8 |
| exp37_fs_loc | 16/16 | 1/1 | 16/16 | 4/4 |
| exp37_fs_joint | 16/16 | 3/3 | 48/48 | 12/12 |
| exp37_fs_both | 16/16 | 3/3 | 48/48 | 12/12 |
| 合计 | 64/64 | 9/9 | 144/144 | 36/36 |

zero-shot每项覆盖四图：定位／离散8000样本、连续5120帧；few-shot每个目标图为2000／2000／1280，连续每图20个64帧clip。训练终点400/400，support seed=0、推理seed=42、minimal资产与 `benchmark_v2_core` 指标。已核对推理配置哈希、batch、split、地图、样本数、指标内嵌推理记录及四图宏平均。FID／FVD均为四图逐图指标等权平均，未将四图混池重算；core中未启用的可选指标为null，不作为本轮缺失项。

本轮新增对齐batch的gen／loc／joint zero-shot、loc的16个bs16 few-shot点和4份定位汇总，以及both 20／10-shot的6份正式汇总；旧适配模型被复用，训练点仍为64个。保留的Seen-retention继续暂停，已有bs1 retention不改标为bs16。部分远端绝对路径仍保存在指标或汇总元数据中；内容核对按相同相对输出身份进行，本机原样重跑严格aggregate前可能需要迁移路径，不视为训练、推理或指标缺失。本次仅整理结果，未执行实验。

### 定位：zero-shot与四档few-shot四图等权macro（bs16）

| Family | Shot | 推理batch | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 正式指标／汇总 |
|---|---|---|---|---|---|---|---|
| exp37_fs_loc | 0 | 16 | 340.1490 | 19.8807 | 7.6202 | 91.6243 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_loc | 100 | 16 | 187.7488 | 13.2410 | 4.4524 | 76.9147 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/shot_100/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_loc | 50 | 16 | 201.4446 | 14.2405 | 4.7925 | 82.1323 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/shot_50/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_loc | 20 | 16 | 207.9848 | 14.7999 | 4.8244 | 84.1623 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/shot_20/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_loc | 10 | 16 | 227.1583 | 15.0948 | 5.7164 | 86.4365 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/shot_10/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_joint | 0 | 16 | 344.6976 | 18.7476 | 7.8122 | 92.1011 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_joint | 100 | 16 | 173.7838 | 12.3034 | 4.8829 | 74.5448 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/shot_100/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_joint | 50 | 16 | 193.3055 | 13.3759 | 4.8543 | 81.4052 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/shot_50/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_joint | 20 | 16 | 206.3759 | 14.3912 | 5.3183 | 84.9582 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/shot_20/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_joint | 10 | 16 | 223.9695 | 15.2190 | 5.4017 | 88.5198 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/shot_10/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_both | 0 | 16 | 341.5329 | 18.9907 | 9.0030 | 93.0790 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_both | 100 | 16 | 171.8577 | 11.9646 | 4.8400 | 74.9741 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both/shot_100/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_both | 50 | 16 | 188.9630 | 13.2528 | 5.0187 | 80.8331 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both/shot_50/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_both | 20 | 16 | 206.1598 | 14.3399 | 5.7314 | 85.0548 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both/shot_20/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| exp37_fs_both | 10 | 16 | 222.5440 | 15.1374 | 6.2334 | 88.4703 | [JSON](outputs_loc/benchmark_v2/exp37_fs_both/shot_10/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |

### 离散生成：zero-shot与四档few-shot四图等权macro（bs32）

| Family | Shot | 推理batch | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | 正式指标／汇总 |
|---|---|---|---|---|---|---|---|---|---|
| exp37_fs_gen | 0 | 32 | 11.3624 | 0.3800 | 0.7168 | 0.4578 | 77.5945 | 0.7458 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/discrete/summary.json) |
| exp37_fs_gen | 100 | 32 | 13.2324 | 0.4350 | 0.6523 | 0.4856 | 37.2678 | 0.8294 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_gen | 50 | 32 | 12.9260 | 0.4277 | 0.6646 | 0.4789 | 39.4927 | 0.8214 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_gen | 20 | 32 | 12.7078 | 0.4266 | 0.6748 | 0.4697 | 43.3891 | 0.8172 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_gen | 10 | 32 | 12.5778 | 0.4218 | 0.6806 | 0.4621 | 48.1832 | 0.8084 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_joint | 0 | 32 | 11.0401 | 0.3730 | 0.7281 | 0.4604 | 104.1529 | 0.7328 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/discrete/summary.json) |
| exp37_fs_joint | 100 | 32 | 13.0264 | 0.4320 | 0.6681 | 0.4809 | 38.8996 | 0.8145 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_joint | 50 | 32 | 12.8645 | 0.4293 | 0.6757 | 0.4753 | 40.6680 | 0.8089 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_joint | 20 | 32 | 12.6104 | 0.4280 | 0.6807 | 0.4688 | 45.4271 | 0.8054 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_joint | 10 | 32 | 12.5602 | 0.4168 | 0.6845 | 0.4661 | 52.8807 | 0.8049 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_both | 0 | 32 | 11.0518 | 0.3706 | 0.7285 | 0.4604 | 100.0587 | 0.7324 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/zero_shot/seed_42/infer_bs32/discrete/summary.json) |
| exp37_fs_both | 100 | 32 | 13.0418 | 0.4293 | 0.6673 | 0.4807 | 39.5391 | 0.8154 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_both | 50 | 32 | 12.8144 | 0.4265 | 0.6747 | 0.4761 | 40.3697 | 0.8124 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_both | 20 | 32 | 12.5950 | 0.4255 | 0.6826 | 0.4687 | 45.3569 | 0.8053 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| exp37_fs_both | 10 | 32 | 12.4688 | 0.4152 | 0.6883 | 0.4641 | 51.5688 | 0.8022 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |

### 连续生成：zero-shot与四档few-shot四图等权macro（bs32）

| Family | Shot | 推理batch | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 正式指标／汇总 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp37_fs_gen | 0 | 32 | 11.3340 | 0.3731 | 0.7167 | 0.4573 | 44.7144 | 50.3376 | 16.8801 | 1286.7474 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/continuous/summary.json) |
| exp37_fs_gen | 100 | 32 | 13.0882 | 0.4202 | 0.6553 | 0.4873 | 37.9702 | 44.0275 | 16.6461 | 944.1510 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_gen | 50 | 32 | 12.5712 | 0.4091 | 0.6755 | 0.4746 | 38.7101 | 44.7269 | 16.6252 | 919.7629 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_gen | 20 | 32 | 12.3739 | 0.4088 | 0.6791 | 0.4679 | 36.7182 | 43.0405 | 16.7315 | 887.8895 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_gen | 10 | 32 | 12.2559 | 0.4021 | 0.6863 | 0.4634 | 36.6729 | 42.9984 | 17.1497 | 935.6509 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_joint | 0 | 32 | 11.1151 | 0.3675 | 0.7280 | 0.4575 | 42.3915 | 48.3329 | 16.7690 | 1358.0126 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/continuous/summary.json) |
| exp37_fs_joint | 100 | 32 | 12.8284 | 0.4135 | 0.6735 | 0.4836 | 40.8774 | 46.6499 | 16.4464 | 1007.6077 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_joint | 50 | 32 | 12.4957 | 0.4075 | 0.6837 | 0.4752 | 40.3112 | 46.2875 | 16.4986 | 1010.9113 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_joint | 20 | 32 | 12.3564 | 0.4103 | 0.6856 | 0.4701 | 38.2192 | 44.3905 | 16.5277 | 960.9629 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_joint | 10 | 32 | 12.3481 | 0.3986 | 0.6904 | 0.4694 | 35.9603 | 42.3631 | 16.6376 | 964.6313 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_both | 0 | 32 | 11.1011 | 0.3657 | 0.7289 | 0.4581 | 43.0778 | 48.9258 | 16.9266 | 1334.9445 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/zero_shot/seed_42/infer_bs32/continuous/summary.json) |
| exp37_fs_both | 100 | 32 | 12.8341 | 0.4119 | 0.6730 | 0.4846 | 40.3321 | 46.2007 | 16.3582 | 984.8784 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_both | 50 | 32 | 12.4543 | 0.4072 | 0.6850 | 0.4765 | 40.4484 | 46.3713 | 16.4332 | 1030.1365 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_both | 20 | 32 | 12.3794 | 0.4088 | 0.6889 | 0.4725 | 38.7084 | 44.8006 | 16.5945 | 950.1136 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| exp37_fs_both | 10 | 32 | 12.2896 | 0.3971 | 0.6932 | 0.4662 | 37.4045 | 43.6475 | 16.7706 | 933.5771 | [JSON](outputs_eval/benchmark_v2/exp37_fs_both/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |

### 本轮补齐的loc单任务逐图结果（bs16）

| 目标地图 | Shot | 推理batch | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 原始指标 |
|---|---|---|---|---|---|---|---|
| cs_office | 100 | 16 | 111.6945 | 2.7938 | 4.3538 | 74.4326 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 50 | 16 | 135.9890 | 3.2913 | 4.6051 | 83.7286 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 20 | 16 | 151.8379 | 3.4999 | 4.9081 | 81.7366 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 10 | 16 | 169.0364 | 3.7493 | 5.5096 | 84.8408 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_cs_office/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 100 | 16 | 196.3496 | 13.9362 | 3.5985 | 79.2652 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 50 | 16 | 200.3117 | 14.2438 | 4.3488 | 84.6923 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 20 | 16 | 220.3549 | 15.1682 | 3.9401 | 87.6140 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 10 | 16 | 249.4312 | 15.6833 | 4.2235 | 86.3630 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_golden/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 100 | 16 | 255.3654 | 16.6891 | 3.7013 | 71.6113 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_palacio/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 50 | 16 | 273.0626 | 18.5710 | 3.5252 | 78.0826 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_palacio/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 20 | 16 | 262.8679 | 18.7117 | 3.2886 | 81.5098 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_palacio/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 10 | 16 | 283.5926 | 18.2623 | 3.5586 | 87.0620 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_palacio/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 100 | 16 | 187.5857 | 19.5448 | 6.1561 | 82.3495 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_vertigo/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 50 | 16 | 196.4150 | 20.8559 | 6.6908 | 82.0258 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_vertigo/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 20 | 16 | 196.8785 | 21.8198 | 7.1607 | 85.7887 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_vertigo/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 10 | 16 | 206.5732 | 22.6844 | 9.5738 | 87.4803 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc_de_vertigo/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |

### 本轮补齐的定位zero-shot逐图结果（bs16）

| Family | 目标地图 | 推理batch | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 原始指标 |
|---|---|---|---|---|---|---|---|
| exp37_fs_loc | cs_office | 16 | 439.6088 | 15.0878 | 7.4960 | 95.1125 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_loc | de_golden | 16 | 286.9377 | 17.9451 | 5.3357 | 90.2417 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_loc | de_palacio | 16 | 363.6545 | 21.3622 | 3.6463 | 91.1925 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_loc | de_vertigo | 16 | 270.3950 | 25.1276 | 14.0027 | 89.9506 | [JSON](outputs_loc/benchmark_v2/exp37_fs_loc/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_joint | cs_office | 16 | 450.6829 | 11.2930 | 8.3085 | 91.3564 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_joint | de_golden | 16 | 279.0272 | 18.1757 | 7.0636 | 94.1394 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_joint | de_palacio | 16 | 370.8823 | 21.1277 | 4.8019 | 88.0758 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |
| exp37_fs_joint | de_vertigo | 16 | 278.1982 | 24.3939 | 11.0748 | 94.8327 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs16/benchmark_csgo_v2_loc.json) |

### 本轮补齐的离散生成zero-shot逐图结果（bs32）

| Family | 目标地图 | 推理batch | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | 原始指标 |
|---|---|---|---|---|---|---|---|---|---|
| exp37_fs_gen | cs_office | 32 | 11.3174 | 0.4706 | 0.7257 | 0.4260 | 88.9933 | 0.7596 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| exp37_fs_gen | de_golden | 32 | 11.6874 | 0.3321 | 0.6871 | 0.4819 | 54.4190 | 0.7529 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| exp37_fs_gen | de_palacio | 32 | 11.3244 | 0.3292 | 0.7332 | 0.4706 | 78.6537 | 0.7429 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| exp37_fs_gen | de_vertigo | 32 | 11.1204 | 0.3880 | 0.7212 | 0.4526 | 88.3120 | 0.7279 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| exp37_fs_joint | cs_office | 32 | 11.1526 | 0.4377 | 0.7446 | 0.4446 | 126.2218 | 0.7422 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| exp37_fs_joint | de_golden | 32 | 11.4677 | 0.3419 | 0.6808 | 0.4853 | 88.3751 | 0.7439 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| exp37_fs_joint | de_palacio | 32 | 11.3458 | 0.3735 | 0.7321 | 0.4645 | 95.0794 | 0.7430 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| exp37_fs_joint | de_vertigo | 32 | 10.1943 | 0.3390 | 0.7548 | 0.4472 | 106.9352 | 0.7020 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |

### 本轮补齐的连续生成zero-shot逐图结果（bs32）

| Family | 目标地图 | 推理batch | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 原始指标 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp37_fs_gen | cs_office | 32 | 11.1715 | 0.4604 | 0.7430 | 0.4234 | 44.5355 | 50.3330 | 15.8215 | 888.3153 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| exp37_fs_gen | de_golden | 32 | 11.6985 | 0.3192 | 0.6864 | 0.4856 | 43.7345 | 48.1686 | 17.8213 | 1404.7126 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| exp37_fs_gen | de_palacio | 32 | 11.4087 | 0.3236 | 0.7183 | 0.4690 | 43.8790 | 49.9733 | 17.5801 | 1484.8345 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| exp37_fs_gen | de_vertigo | 32 | 11.0574 | 0.3891 | 0.7191 | 0.4512 | 46.7086 | 52.8757 | 16.2975 | 1369.1274 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| exp37_fs_joint | cs_office | 32 | 11.3945 | 0.4321 | 0.7485 | 0.4431 | 32.5244 | 39.4361 | 15.4334 | 1042.8521 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| exp37_fs_joint | de_golden | 32 | 11.8984 | 0.3400 | 0.6848 | 0.4876 | 44.8013 | 49.0050 | 17.9833 | 1507.9604 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| exp37_fs_joint | de_palacio | 32 | 11.0779 | 0.3540 | 0.7310 | 0.4564 | 42.4097 | 48.9828 | 17.0730 | 1426.8051 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| exp37_fs_joint | de_vertigo | 32 | 10.0898 | 0.3440 | 0.7479 | 0.4427 | 49.8306 | 55.9077 | 16.5864 | 1454.4328 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |

<a id="exp37-unseen4-aligned-analysis-20261113"></a>

## 2026-11-13：对齐 batch 后的实验分析

下表是完整曲线中的固定100-shot切片，各项都使用相同的任务推理batch；不根据Unseen query选择shot或配方。单任务两行来自各自模型，不能合称一个联合模型。

| Family | 定位batch／生成batch | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 离散PSNR↑ | 离散LPIPS↓ | FID↓ | 连续PSNR↑ | 连续LPIPS↓ | FVD↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp37_fs_gen | —／32 | N/A | N/A | N/A | N/A | 13.2324 | 0.6523 | 37.2678 | 13.0882 | 0.6553 | 944.1510 |
| exp37_fs_loc | 16／— | 187.7488 | 13.2410 | 4.4524 | 76.9147 | N/A | N/A | N/A | N/A | N/A | N/A |
| exp37_fs_joint | 16／32 | 173.7838 | 12.3034 | 4.8829 | 74.5448 | 13.0264 | 0.6681 | 38.8996 | 12.8284 | 0.6735 | 1007.6077 |
| exp37_fs_both | 16／32 | 171.8577 | 11.9646 | 4.8400 | 74.9741 | 13.0418 | 0.6673 | 39.5391 | 12.8341 | 0.6730 | 984.8784 |

1. **XY四档均呈both < joint < loc单任务，整体定位仍有维度取舍。** 100-shot XY为171.8577／173.7838／187.7488，joint和both相对loc单任务分别降低7.44%和8.46%；both相对joint的四档XY、Z均较低。但100／50／20-shot的Pitch仍由loc单任务较低，20／10-shot的Yaw也由loc单任务较低；10-shot Z为loc 15.0948、both 15.1374、joint 15.2190。因此位置误差支持该排序，不能推广成全部5D指标均满足该排序。
2. **生成单任务的主要图像指标优势仍然存在。** 四档生成bs32下，gen-only的离散PSNR、LPIPS、FID和CLIP都优于joint，连续LPIPS、FVD也均优于joint。100-shot离散FID为gen 37.2678、joint 38.8996、both 39.5391，连续FVD为944.1510／1007.6077／984.8784。both相对joint并非全面提升：50-shot连续FVD反而更高；10-shot both的FVD低于gen-only，但逐帧LPIPS更高。joint的连续OFE在四档均低于gen-only，不能由单项时序指标推断生成整体更好。
3. **同batch zero-shot已齐，可以计算各自来源的适配收益。** 100-shot相对本family zero-shot，XY降幅为loc 44.80%、joint 49.58%、both 49.68%；离散FID降幅为gen 51.97%、joint 62.65%、both 60.48%；连续FVD降幅为gen 26.63%、joint 25.80%、both 26.22%。这些是各自不同来源上的适配收益；较大的相对降幅不等于适配后的绝对指标更好，也不单独证明辅助loss有效。旧bs1 zero-shot不参与本组百分比计算。
4. **增加shot改善位置和逐帧质量，时序指标并不单调。** loc单任务的四项误差随10→20→50→100-shot依次降低；joint与both的XY亦依次下降。gen／joint的连续FVD在20-shot低于100-shot，both则10-shot低于100-shot。各点虽为400次更新，单任务曝光为4k／8k／20k／40k，joint／both为3.2k／8k／19.2k／40k；保持既定shot曲线，不把它解释为等FLOPs比较或据query重新选点。
5. **当前结论与证据边界。** batch差异已从当前主表排除，但单任务来源的历史训练预算、paired任务曝光，以及joint／both在Seen Stage2和few-shot阶段均不同的目标仍然存在。当前证据支持联合模型的位置误差收益及辅助loss的部分改进，尚不支持“辅助loss + 联合训练 > 联合训练 > 单任务”的全面排序；也不能把whole-pipeline差异只归因于few-shot辅助项。四个选定family的zero-shot、four-shot曲线和正式汇总均无缺失；both_b1不列入缺口，retention继续暂停，不宣称没有Seen遗忘。

## 已同步 zero-shot 结果（2026-11-07 补充）

本节为2026-11-07历史快照：gen／loc／joint为bs1，both为loc16／gen32。最新四个family均已对齐batch，见[当前完整结果](#exp37-unseen4-aligned-results-20261113)；下表不与新结果混算。

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

**以下2026-11-06完成结果均使用batch=1（历史结果）**，与对应family的zero-shot一致；不是新增默认定位32／生成16的结果。每张地图使用自己的适配模型，下面是四图等权宏平均，FID／FVD也是四个逐图指标的平均，不是把所有图片／视频混池重算。正式四图`map-models`聚合JSON尚未同步，表中macro从四个单图JSON直接复算；下方逐图表提供全部源链接。

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

<a id="exp37-joint-results-20261112"></a>

## 2026-11-12：exp37_fs_joint 完整 few-shot query 结果

本节保留2026-11-12核查时点；当时缺少的同batch zero-shot及loc单任务对照现已补齐，当前比较见[2026-11-13结果](#exp37-unseen4-aligned-results-20261113)与[分析](#exp37-unseen4-aligned-analysis-20261113)。

四张地图×100／50／20／10-shot的16个独立适配点均完成local400/400，直接来源为 `exp37_s2@6000`，固定 αloc=8、无perception／aux-loc。定位batch16、离散／连续生成batch32共48份单图query指标与12份正式 `map-models` 汇总均已同步。support seed=0、推理seed=42、minimal资产；每点定位／离散为2000样本，连续为1280帧（20个64帧clip），每档四图汇总分别为8000／8000／5120样本。所有宏平均与四图原始JSON复算一致，FID／FVD也是逐图指标等权平均，未将四图混池重算。

此前100／50-shot缺失的office结果已补齐；100-shot office三类指标中的旧服务器路径已迁移，正式汇总现已生成。本次读取的是修复后的canonical JSON，不将备份文件计为额外结果；未重算指标或修改训练／推理配置。joint zero-shot仍只有历史batch1；本节不计算跨batch的zero-shot增益。Seen-retention未补跑，继续暂停。

**跨服务器路径备注：** 20／10-shot的24份逐图指标及6份正式汇总仍保留原服务器根路径 `/home/user/yc57963/task/UniLIP`。本次核对资产报告哈希、报告内的benchmark manifest哈希与本机一致，指标内嵌的推理记录与同目录manifest一致，再仅对内存副本映射路径完成来源与数值核验；原JSON未修改。这些结果与四图macro已齐全，但在本机原样再次执行严格 `aggregate` 仍可能因旧绝对路径失败，需要先迁移路径元数据，无需据此重新训练或推理。100-shot office生成指标的 `gt_dir`／`pred_dir` 仍保留原运行位置作为记录，不影响已完成的本次来源校验。

### 定位：四图等权 macro（batch16）

| Shot | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 正式 map-models 汇总 |
|---|---|---|---|---|---|
| 100-shot | 173.7838 | 12.3034 | 4.8829 | 74.5448 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/shot_100/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| 50-shot | 193.3055 | 13.3759 | 4.8543 | 81.4052 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/shot_50/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| 20-shot | 206.3759 | 14.3912 | 5.3183 | 84.9582 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/shot_20/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |
| 10-shot | 223.9695 | 15.2190 | 5.4017 | 88.5198 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint/shot_10/seed_0/infer_bs16/map_models/benchmark_v2_localization_crossmap.json) |

### 定位：逐图结果（batch16）

| 目标地图 | Shot | XY↓ | Z↓ | Pitch↓ | Yaw↓ | 原始指标 |
|---|---|---|---|---|---|---|
| cs_office | 100 | 100.0794 | 2.4092 | 5.4638 | 73.0420 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_cs_office/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 50 | 132.2481 | 3.0081 | 5.6109 | 86.6590 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_cs_office/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 20 | 142.0083 | 3.4553 | 6.1692 | 90.0868 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_cs_office/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| cs_office | 10 | 149.2420 | 3.8008 | 5.6893 | 92.2777 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_cs_office/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 100 | 186.3652 | 14.1861 | 4.0663 | 79.6353 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_golden/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 50 | 194.5760 | 14.8489 | 4.5106 | 84.7603 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_golden/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 20 | 216.7346 | 16.0052 | 4.4605 | 86.5182 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_golden/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_golden | 10 | 237.2417 | 16.8112 | 4.8991 | 86.2766 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_golden/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 100 | 234.7656 | 14.8866 | 4.0670 | 66.5326 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_palacio/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 50 | 258.5663 | 16.2010 | 3.7933 | 72.0780 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_palacio/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 20 | 269.4518 | 16.2584 | 4.3630 | 77.6260 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_palacio/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_palacio | 10 | 295.1040 | 17.1853 | 4.1359 | 85.1034 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_palacio/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 100 | 173.9249 | 17.7316 | 5.9346 | 78.9692 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_vertigo/shot_100/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 50 | 187.8316 | 19.4457 | 5.5024 | 82.1235 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_vertigo/shot_50/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 20 | 197.3089 | 21.8458 | 6.2807 | 85.6018 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_vertigo/shot_20/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |
| de_vertigo | 10 | 214.2905 | 23.0788 | 6.8827 | 90.4214 | [JSON](outputs_loc/benchmark_v2/exp37_fs_joint_de_vertigo/shot_10/seed_0/infer_bs16/benchmark_csgo_v2_loc.json) |

### 离散生成：四图等权 macro（batch32）

| Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | 正式 map-models 汇总 |
|---|---|---|---|---|---|---|---|
| 100-shot | 13.0264 | 0.4320 | 0.6681 | 0.4809 | 38.8996 | 0.8145 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| 50-shot | 12.8645 | 0.4293 | 0.6757 | 0.4753 | 40.6680 | 0.8089 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| 20-shot | 12.6104 | 0.4280 | 0.6807 | 0.4688 | 45.4271 | 0.8054 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| 10-shot | 12.5602 | 0.4168 | 0.6845 | 0.4661 | 52.8807 | 0.8049 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |

### 离散生成：逐图结果（batch32）

| 目标地图 | Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | 原始指标 |
|---|---|---|---|---|---|---|---|---|
| cs_office | 100 | 13.2928 | 0.5674 | 0.6490 | 0.4460 | 39.0351 | 0.8472 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_cs_office/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 50 | 13.1729 | 0.5650 | 0.6634 | 0.4396 | 42.3528 | 0.8364 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_cs_office/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 20 | 12.6177 | 0.5714 | 0.6699 | 0.4282 | 52.0593 | 0.8368 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_cs_office/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 10 | 12.9521 | 0.5524 | 0.6761 | 0.4205 | 58.1776 | 0.8277 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_cs_office/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| de_golden | 100 | 14.4698 | 0.3664 | 0.6343 | 0.5172 | 39.5099 | 0.8005 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_golden/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 50 | 14.3273 | 0.3608 | 0.6370 | 0.5167 | 38.8082 | 0.7972 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_golden/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 20 | 14.2308 | 0.3542 | 0.6412 | 0.5098 | 42.9454 | 0.7970 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_golden/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 10 | 14.1380 | 0.3473 | 0.6465 | 0.5045 | 47.5582 | 0.8106 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_golden/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_palacio | 100 | 12.5736 | 0.3532 | 0.7099 | 0.5096 | 37.7459 | 0.8020 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_palacio/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 50 | 12.3218 | 0.3530 | 0.7228 | 0.4983 | 41.8597 | 0.7923 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_palacio/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 20 | 12.1113 | 0.3412 | 0.7197 | 0.4989 | 43.0281 | 0.7966 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_palacio/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 10 | 11.6135 | 0.3077 | 0.7224 | 0.4945 | 52.4353 | 0.7896 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_palacio/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_vertigo | 100 | 11.7693 | 0.4411 | 0.6793 | 0.4509 | 39.3074 | 0.8084 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_vertigo/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 50 | 11.6359 | 0.4383 | 0.6795 | 0.4465 | 39.6515 | 0.8095 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_vertigo/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 20 | 11.4820 | 0.4451 | 0.6919 | 0.4381 | 43.6756 | 0.7912 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_vertigo/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 10 | 11.5374 | 0.4598 | 0.6932 | 0.4449 | 53.3515 | 0.7917 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_vertigo/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |

### 连续生成：四图等权 macro（batch32）

| Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 正式 map-models 汇总 |
|---|---|---|---|---|---|---|---|---|---|
| 100-shot | 12.8284 | 0.4135 | 0.6735 | 0.4836 | 40.8774 | 46.6499 | 16.4464 | 1007.6077 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| 50-shot | 12.4957 | 0.4075 | 0.6837 | 0.4752 | 40.3112 | 46.2875 | 16.4986 | 1010.9113 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| 20-shot | 12.3564 | 0.4103 | 0.6856 | 0.4701 | 38.2192 | 44.3905 | 16.5277 | 960.9629 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| 10-shot | 12.3481 | 0.3986 | 0.6904 | 0.4694 | 35.9603 | 42.3631 | 16.6376 | 964.6313 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |

### 连续生成：逐图结果（batch32）

| 目标地图 | Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 原始指标 |
|---|---|---|---|---|---|---|---|---|---|---|
| cs_office | 100 | 12.9857 | 0.5297 | 0.6675 | 0.4503 | 40.7560 | 46.2787 | 14.8583 | 794.0831 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_cs_office/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 50 | 12.6101 | 0.5230 | 0.6905 | 0.4332 | 39.8758 | 45.8001 | 14.7305 | 858.3618 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_cs_office/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 20 | 12.1705 | 0.5322 | 0.6861 | 0.4370 | 36.4907 | 42.7120 | 14.5784 | 739.6609 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_cs_office/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 10 | 12.7811 | 0.5165 | 0.6962 | 0.4235 | 35.5334 | 41.8977 | 15.3677 | 758.4152 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_cs_office/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| de_golden | 100 | 14.4753 | 0.3598 | 0.6320 | 0.5228 | 33.7659 | 38.9026 | 17.6136 | 1205.6675 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_golden/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 50 | 14.2590 | 0.3530 | 0.6343 | 0.5226 | 33.7106 | 38.8678 | 17.8754 | 1193.5360 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_golden/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 20 | 14.1927 | 0.3496 | 0.6383 | 0.5141 | 31.1388 | 36.5741 | 17.8012 | 1096.5546 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_golden/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 10 | 13.9466 | 0.3380 | 0.6423 | 0.5159 | 31.4093 | 36.7401 | 17.8544 | 1032.1474 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_golden/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_palacio | 100 | 12.3363 | 0.3327 | 0.7188 | 0.5112 | 42.1086 | 48.1615 | 17.2472 | 1051.2470 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_palacio/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 50 | 11.8247 | 0.3315 | 0.7335 | 0.5009 | 41.6573 | 47.9304 | 17.2468 | 1017.5259 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_palacio/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 20 | 12.0951 | 0.3338 | 0.7196 | 0.4999 | 40.7755 | 47.0243 | 17.7714 | 1068.3486 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_palacio/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 10 | 11.7107 | 0.3045 | 0.7222 | 0.4989 | 39.6549 | 46.0807 | 18.2487 | 1076.3709 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_palacio/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_vertigo | 100 | 11.5163 | 0.4320 | 0.6756 | 0.4504 | 46.8791 | 53.2569 | 16.0664 | 979.4330 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_vertigo/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 50 | 11.2889 | 0.4226 | 0.6765 | 0.4443 | 46.0009 | 52.5517 | 16.1418 | 974.2217 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_vertigo/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 20 | 10.9671 | 0.4254 | 0.6986 | 0.4294 | 44.4719 | 51.2516 | 15.9596 | 939.2875 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_vertigo/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 10 | 10.9538 | 0.4356 | 0.7008 | 0.4395 | 37.2438 | 44.7338 | 15.0794 | 991.5916 | [JSON](outputs_eval/benchmark_v2/exp37_fs_joint_de_vertigo/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |

<a id="exp37-joint-analysis-20261112"></a>

## 2026-11-12：joint、both 与单任务的结果分析

以下为当时的历史分析；涉及loc单任务尚未对齐、zero-shot缺失的限制已由[2026-11-13结果](#exp37-unseen4-aligned-results-20261113)更新，最新结论见[对齐batch分析](#exp37-unseen4-aligned-analysis-20261113)。

以下先比较joint与both的匹配推理批量：定位16、生成32；训练均为每图独立400次更新、support seed=0、相同冻结query。表中为 **both − joint**，各列均越低越好，负值表示both较低；both数值来自其已有逐图指标及[完整结果](#exp37-both-results-20261109)。

| Shot | ΔXY↓ | ΔZ↓ | ΔPitch↓ | ΔYaw↓ | 离散ΔLPIPS↓ | ΔFID↓ | 连续ΔLPIPS↓ | ΔFVD↓ |
|---|---|---|---|---|---|---|---|---|
| 100 | -1.9260 | -0.3388 | -0.0429 | +0.4293 | -0.0009 | +0.6395 | -0.0005 | -22.7292 |
| 50 | -4.3425 | -0.1232 | +0.1644 | -0.5721 | -0.0010 | -0.2983 | +0.0013 | +19.2251 |
| 20 | -0.2161 | -0.0512 | +0.4131 | +0.0966 | +0.0019 | -0.0702 | +0.0033 | -10.8493 |
| 10 | -1.4255 | -0.0816 | +0.8317 | -0.0495 | +0.0037 | -1.3119 | +0.0029 | -31.0542 |

1. **joint随shot增加改善位置误差和逐帧质量，但并非每项指标单调。** 从10→20→50→100-shot，XY为223.9695→206.3759→193.3055→173.7838，Z、Yaw亦依次下降；离散PSNR由12.5602升至13.0264，FID由52.8807降至38.8996。Pitch在50-shot为4.8543，100-shot为4.8829；连续FVD依次为964.6313／960.9629／1010.9113／1007.6077，不随shot单调改善。各点虽固定400次更新，joint base样本曝光分别为3200／8000／19200／40000，不能把趋势只归因于独立support数量，也不能称等FLOPs。
2. **both的位置误差有一致的小幅改善，朝向误差存在取舍。** 四档XY和Z均低于joint；XY相对降幅约0.10%～2.25%，100-shot由173.7838降至171.8577。Pitch仅100-shot略低，其余三档更高；Yaw仅50／10-shot更低。因此当前结果支持位置误差改善，不支持所有定位维度都改善。
3. **辅助loss没有带来生成指标的全面优势。** both的离散FID在50／20／10-shot较低，100-shot反而由38.8996升至39.5391；离散LPIPS在100／50-shot较低，在20／10-shot较高。连续FVD在100／20／10-shot较低，50-shot由1010.9113升至1030.1365；连续LPIPS只在100-shot较低。100-shot的部分生成收益不能概括整个few-shot曲线，也不能仅凭FVD推断逐帧图像质量提升。
4. **同batch的gen-only仍保留生成优势，但也有时序取舍。** 与[gen batch32结果](#exp37-gen-bs32-results-20261111)相比，joint四档的离散PSNR、LPIPS、FID、CLIP均较差，连续LPIPS与FVD也均较差；100-shot离散FID为38.8996 vs 37.2678，连续FVD为1007.6077 vs 944.1510。不过joint四档连续OFE均较低，10-shot连续PSNR也较高。loc-only现有结果仍为batch1，本节不据其与joint batch16的差值判断联合训练提升。
5. **比较边界与下一步。** joint与both共享Seen Stage1来源，但Seen Stage2已分别训练为无辅助／有辅助模型，few-shot又沿用各自方案；本轮比较的是完整训练流程，不能单独归因于few-shot阶段的辅助loss。gen-only来源历史和任务曝光也不同。当前不支持“辅助loss + 联合训练 > 联合训练 > 单任务”的全面排序。保留冻结配方与全部shot，优先完善loc单任务batch16、joint的loc16／gen32及gen的gen32 zero-shot对照；不依据Unseen query重新调参、挑选shot或反选both候选。Seen-retention继续暂停，不宣称适配后保持Seen能力。

本次只整理已完成实验，未启动训练、推理、指标重算或额外聚合。

<a id="exp37-gen-bs32-results-20261111"></a>

## 2026-11-11：exp37_fs_gen batch32 few-shot 生成 query

本节为2026-11-11快照；当时未同步的bs32 zero-shot现已补齐，见[2026-11-13完整曲线](#exp37-unseen4-aligned-results-20261113)。下列few-shot数值保持原样。

原有四图四档共16个独立 final400 适配模型在 `infer_bs32` 下完成离散和连续生成：32/32项推理、32/32份逐图指标及8/8份正式 `map-models` 汇总均已同步。`cs_office` 50-shot先前未完成的离散／连续两项现也齐全。每点离散覆盖2000张、连续覆盖1280帧（20个64帧clip）；逐图预测覆盖完整，缺失和多余预测均为0。四档汇总均为四个目标地图独立模型的等权宏平均，含FID／FVD；8份正式汇总与逐图JSON复算在数值精度内一致。使用原有适配模型，不新增训练；support seed=0、推理seed=42、minimal资产、`benchmark_v2_core`指标口径。生成清单记录438个active Dropout、p=0.05及`stochastic_layers_batch_dependent=true`。本节只整理已同步产物，未重新训练、推理或评测。

这里的0-shot gen只有历史batch1结果，尚无batch32 zero-shot，故不能以本节few-shot batch32数值计算同batch zero-shot增益。旧batch1的宏平均与逐图表仍在[历史结果](#exp37-fewshot-results-20261106)；不同batch的值分开报告。Seen-retention继续暂停。

### 离散生成：四图等权 macro（batch32）

| Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | 正式 map-models 汇总 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100-shot | 13.2324 | 0.4350 | 0.6523 | 0.4856 | 37.2678 | 0.8294 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| 50-shot | 12.9260 | 0.4277 | 0.6646 | 0.4789 | 39.4927 | 0.8214 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| 20-shot | 12.7078 | 0.4266 | 0.6748 | 0.4697 | 43.3891 | 0.8172 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |
| 10-shot | 12.5778 | 0.4218 | 0.6806 | 0.4621 | 48.1832 | 0.8084 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_discrete_crossmap.json) |

### 离散生成：逐图结果（batch32）

| 目标地图 | Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ | CLIP↑ | 原始指标 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cs_office | 100 | 13.5905 | 0.5762 | 0.6301 | 0.4561 | 36.2645 | 0.8584 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 50 | 13.2888 | 0.5725 | 0.6544 | 0.4434 | 38.7519 | 0.8484 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 20 | 12.6731 | 0.5694 | 0.6713 | 0.4329 | 44.9264 | 0.8397 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| cs_office | 10 | 12.5748 | 0.5686 | 0.6797 | 0.4178 | 50.2148 | 0.8198 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_cs_office.json) |
| de_golden | 100 | 14.6886 | 0.3691 | 0.6202 | 0.5223 | 38.3966 | 0.8228 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 50 | 14.6072 | 0.3645 | 0.6285 | 0.5158 | 37.4586 | 0.8131 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 20 | 14.3750 | 0.3613 | 0.6340 | 0.5078 | 39.2944 | 0.8149 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_golden | 10 | 14.2433 | 0.3444 | 0.6344 | 0.5031 | 47.9766 | 0.8187 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_golden.json) |
| de_palacio | 100 | 12.6907 | 0.3559 | 0.6955 | 0.5100 | 36.5948 | 0.8163 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 50 | 12.1691 | 0.3441 | 0.7026 | 0.5062 | 41.5799 | 0.8120 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 20 | 12.2309 | 0.3391 | 0.7106 | 0.4991 | 42.6595 | 0.8119 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_palacio | 10 | 12.0406 | 0.3197 | 0.7159 | 0.4870 | 46.9016 | 0.8011 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_palacio.json) |
| de_vertigo | 100 | 11.9600 | 0.4389 | 0.6635 | 0.4541 | 37.8152 | 0.8200 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_100/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 50 | 11.6387 | 0.4299 | 0.6731 | 0.4501 | 40.1804 | 0.8119 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_50/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 20 | 11.5520 | 0.4367 | 0.6833 | 0.4391 | 46.6760 | 0.8022 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_20/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |
| de_vertigo | 10 | 11.4524 | 0.4547 | 0.6925 | 0.4407 | 47.6397 | 0.7940 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_10/seed_0/infer_bs32/discrete/benchmark_csgo_v2_de_vertigo.json) |

### 连续生成：四图等权 macro（batch32）

| Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 正式 map-models 汇总 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100-shot | 13.0882 | 0.4202 | 0.6553 | 0.4873 | 37.9702 | 44.0275 | 16.6461 | 944.1510 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_100/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| 50-shot | 12.5712 | 0.4091 | 0.6755 | 0.4746 | 38.7101 | 44.7269 | 16.6252 | 919.7629 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_50/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| 20-shot | 12.3739 | 0.4088 | 0.6791 | 0.4679 | 36.7182 | 43.0405 | 16.7315 | 887.8895 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_20/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |
| 10-shot | 12.2559 | 0.4021 | 0.6863 | 0.4634 | 36.6729 | 42.9984 | 17.1497 | 935.6509 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen/shot_10/seed_0/infer_bs32/map_models/benchmark_v2_continuous_crossmap.json) |

### 连续生成：逐图结果（batch32）

| 目标地图 | Shot | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | T-Warp↓ | T-Diff↓ | OFE↓ | FVD↓ | 原始指标 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cs_office | 100 | 13.6420 | 0.5476 | 0.6414 | 0.4657 | 34.8775 | 40.9379 | 14.4494 | 693.4063 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 50 | 12.7149 | 0.5334 | 0.6852 | 0.4421 | 36.5121 | 42.6090 | 14.5441 | 623.3088 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 20 | 12.3091 | 0.5324 | 0.6925 | 0.4354 | 35.5953 | 41.8759 | 15.2633 | 700.5333 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| cs_office | 10 | 12.0976 | 0.5268 | 0.7087 | 0.4188 | 38.1141 | 44.2394 | 16.4158 | 809.2926 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_cs_office/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_cs_office.json) |
| de_golden | 100 | 14.6489 | 0.3639 | 0.6169 | 0.5301 | 31.9428 | 37.1629 | 17.7630 | 1220.2985 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 50 | 14.6636 | 0.3592 | 0.6277 | 0.5176 | 30.7101 | 36.1564 | 17.6372 | 1126.7824 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 20 | 14.2886 | 0.3542 | 0.6312 | 0.5144 | 29.0214 | 34.7105 | 17.5398 | 1009.1210 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_golden | 10 | 14.1965 | 0.3391 | 0.6296 | 0.5172 | 29.1456 | 34.8476 | 17.9918 | 969.1292 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_golden/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_golden.json) |
| de_palacio | 100 | 12.4278 | 0.3390 | 0.7004 | 0.5003 | 41.1209 | 47.4529 | 17.7084 | 1046.2710 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 50 | 11.7339 | 0.3300 | 0.7152 | 0.4956 | 43.9601 | 49.8242 | 18.1281 | 1052.2097 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 20 | 11.9966 | 0.3310 | 0.7066 | 0.4930 | 39.7961 | 46.1442 | 18.2268 | 982.0384 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_palacio | 10 | 11.6437 | 0.3068 | 0.7097 | 0.4871 | 40.8932 | 47.1051 | 18.9669 | 1050.4742 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_palacio/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_palacio.json) |
| de_vertigo | 100 | 11.6341 | 0.4302 | 0.6627 | 0.4532 | 43.9398 | 50.5564 | 16.6636 | 816.6283 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_100/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 50 | 11.1723 | 0.4137 | 0.6740 | 0.4431 | 43.6582 | 50.3180 | 16.1915 | 876.7507 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_50/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 20 | 10.9014 | 0.4178 | 0.6861 | 0.4289 | 42.4600 | 49.4315 | 15.8962 | 859.8655 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_20/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |
| de_vertigo | 10 | 11.0860 | 0.4355 | 0.6973 | 0.4306 | 38.5389 | 45.8015 | 15.2242 | 913.7075 | [JSON](outputs_eval/benchmark_v2/exp37_fs_gen_de_vertigo/shot_10/seed_0/infer_bs32/continuous/benchmark_csgo_v2_conti_de_vertigo.json) |

### 结果解读

离散生成随shot从10→20→50→100增加，表内六项宏指标逐项改善：PSNR由12.5778升至13.2324，FID由48.1832降至37.2678（22.65%）。连续PSNR、SSIM、LPIPS和Boundary F1也依次改善；FVD按10／20／50／100-shot为935.6509／887.8895／919.7629／944.1510，呈非单调变化。不能根据Unseen query重新选择shot或调整已冻结配方。

与`exp37_fs_both`同为生成batch32的100-shot作描述性比较，gen-only离散PSNR为13.2324 vs 13.0418、FID为37.2678 vs 39.5391，连续FVD为944.1510 vs 984.8784；连续OFE则为16.6461 vs 16.3582（both较低）。同批量消除了这一项推理口径差异，但两组来源训练历史、适配任务和样本曝光不同，不能由这些差值推断辅助loss的因果效果。`exp37_fs_joint`未纳入2026-11-11这次核对，2026-11-12补齐后的比较见[最新分析](#exp37-joint-analysis-20261112)。

<a id="exp37-both-results-20261107"></a>
<a id="exp37-both-results-20261109"></a>

## 2026-11-09：exp37_fs_both（P4＋U2）完整结果与同步状态

2026-11-13补充：四档三类共12份正式汇总现已齐全，20／10-shot的6份新汇总与本节原单图复算值一致；正式来源已全部列入[当前完整结果](#exp37-unseen4-aligned-results-20261113)。以下同步说明保留2026-11-09时点。

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

以下保留2026-11-09的核查快照；其中joint缺失状态已由[2026-11-12完整结果](#exp37-joint-results-20261112)更新。

- **query结果无缺失**：16/16适配点、48/48份单图指标；100／50-shot正式 `map-models` 三项各齐。10／20-shot正式汇总JSON尚未同步，本页macro直接由四图单图JSON等权复算。
- 正式训练权重／Trainer状态已全部同步并核验：16/16个both点final400/400，和32个单任务点合计48个训练终点；query指标96份齐全，完成性分别见[进度表](#exp37-both-progress-20261109)。`exp37_fs_joint` 与 `exp37_fs_both_b1` 是独立family，没有本机few-shot完成证据。Seen-retention按用户决定暂停，未有both retention证据。

<a id="exp37-both-analysis-20261109"></a>

## 2026-11-09：both few-shot结果分析

本节为历史分析；joint已于2026-11-12补齐，同batch对照及当前结论见[最新分析](#exp37-joint-analysis-20261112)。

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

2026-11-13当前状态：选定的gen／loc／joint／both已完成对齐batch的zero-shot、四图四档few-shot及全部正式汇总，当前范围内无待补实验，见[完整结果](#exp37-unseen4-aligned-results-20261113)与[分析](#exp37-unseen4-aligned-analysis-20261113)。后续使用现有结果完成论文表格与证据复核，保持冻结配方，不按Unseen query重新挑选shot或候选。both_b1排除在本轮范围外，原规划保留；Seen-retention继续暂停。

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
