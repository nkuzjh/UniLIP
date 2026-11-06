# CSGO Benchmark v2：exp37 Unseen-4 每图独立 few-shot

状态（2026-11-04 本机同步快照）：**三个来源的 Unseen-4 zero-shot 已完成对应任务评测；80 个每图独立 few-shot 适配点尚无已同步的训练或评测证据**。已完成的是 `exp37_fs_gen` 的两类生成、`exp37_fs_loc` 的定位，以及 `exp37_fs_joint` 的三类任务；`exp37_fs_both` 和 `exp37_fs_both_b1` 的 Stage2@6000 来源仍未同步。本协议承接完整 Seen-10 Stage2 模型，采用用户于 2026-11-03 确认的每图独立适配设置。它不替换历史结果，不恢复暂停的 `exp33/exp34` 四图共同适配支线。命令入口为 [`scripts/run_exp37_unseen4.py`](scripts/run_exp37_unseen4.py)，机器可读矩阵为 [`csgo_configs/benchmark_v2_exp37_unseen4.yaml`](csgo_configs/benchmark_v2_exp37_unseen4.yaml)，手动执行顺序见 [`record.md`](record.md#exp37-unseen4-fewshot-20261103)。

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

query 不能用于学习率、训练长度、checkpoint或最终辅助候选的选择；所有适配配方在观察Unseen结果前冻结。每个适配模型都回测完整Seen-10，报告其各自retention；不把四个模型的retention误当成同一个模型的成绩。

`aggregate` 对同family同shot的四个目标图结果执行 `map-models` 等地图宏平均，保留map→checkpoint映射。zero-shot使用一个父模型覆盖四图，采用普通 `maps` 聚合；两者统计身份不同。单任务缺失的任务列写N/A，未执行的有效任务写待执行，不写0。

## 资产、来源与中断处理

默认使用 minimal 资产报告 `data/csgo_benchmark_v2/minimal_dataset_report.json`。首次训练使用全量资产时传 `--asset-mode full --data-dir /path/to/preprocessed_data`；只替换物理资产后端，不重新采样。后续推理/评测读取本次保存的resolved配置，不能把minimal/full的输出混用。zero-shot首次infer也须明确选择需要的后端。

首次适配执行前核对来源 `checkpoint-6000/model.safetensors`、`trainer_state.json`、来源根目录 `resolved_config.yaml` 和 `experiment_plan.json`；保存来源身份及权重指纹。缺少来源或实际step不是6000时拒绝执行，不寻找替代checkpoint。配置规划和CPU测试不需要读取真实大模型权重。2026-11-04 本机已同步 `exp37_s2_gen`、`exp37_s2_loc`、`exp37_s2` 的上述四项来源文件，Trainer `global_step=6000`；`exp37_s2_both`、`exp37_s2_both_b1` 仍缺这些来源文件。

本协议没有中途完整optimizer checkpoint，**不支持将中断的400步训练用 `--resume` 接续**。若训练中断，需要人工保留/归档本次不完整目录后，从同一个Seen来源重新运行该适配点。已完成400步而推理/评测中断时，只补对应infer/metrics，不重复train；训练入口拒绝覆盖已有运行。不要假定图像推理具有逐样本断点续跑功能。

## 实验记录表

每个单元格代表一个独立训练及其query、retention评测流程。此表登记每图适配的本机同步状态，未同步不代表远端尚未执行；本机无 `outputs/csgo_1b/exp37_fs*` 适配目录或 local400 checkpoint，不能将已完成的 zero-shot 当成 few-shot。

| Family | 目标地图 | 100-shot | 50-shot | 20-shot | 10-shot | 正式checkpoint |
|---|---|---|---|---|---|---|
| exp37_fs_gen | cs_office | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_gen | de_golden | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_gen | de_palacio | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_gen | de_vertigo | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_loc | cs_office | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_loc | de_golden | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_loc | de_palacio | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_loc | de_vertigo | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_joint | cs_office | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_joint | de_golden | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_joint | de_palacio | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_joint | de_vertigo | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both | cs_office | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both | de_golden | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both | de_palacio | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both | de_vertigo | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both_b1 | cs_office | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both_b1 | de_golden | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both_b1 | de_palacio | 未同步 | 未同步 | 未同步 | 未同步 | local400 |
| exp37_fs_both_b1 | de_vertigo | 未同步 | 未同步 | 未同步 | 未同步 | local400 |

zero-shot五组以及四图宏平均结果同步维护在 [`csgo_benchmark_v2_experiments_results.md`](csgo_benchmark_v2_experiments_results.md#exp37-unseen4-results-20261103)。只有实际完成并核对checkpoint、shot、seed、资产后端和覆盖后才能填写数值。

## 已同步 zero-shot 结果（2026-11-04）

下表直接采用原始 JSON 的 `metrics_macro_map`，即四张 Unseen 地图等权平均。定位的 XY／Z／Pitch／Yaw 均为距离误差（越低越好）；生成的 PSNR／SSIM／CLIP 越高越好，其余列越低越好。N/A 表示该来源不执行此任务。三个来源均使用未适配的 Seen Stage2@6000 权重、推理 seed=42、minimal 资产后端；zero-shot 没有 support seed 或 shot。query 为 `crossmap_query_test`，每图2,000条；continuous 为 `crossmap_continuous`，每图20条×64帧。每项已有四图独立指标、推理清单、评测权重计划与聚合 JSON；生成每图 `Common_Count` 分别为2,000／1,280，continuous 每图 `Track_Count=20`。

| 来源／zero-shot | 定位 XY↓ | Z↓ | Pitch↓ | Yaw↓ | 离散 PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | CLIP↑ | 连续 PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp37_fs_gen` | N/A | N/A | N/A | N/A | 11.379 | 0.380 | 0.717 | 77.496 | 0.746 | 11.355 | 0.374 | 0.717 | 1283.955 |
| `exp37_fs_loc` | 339.928 | 19.874 | 7.610 | 91.572 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `exp37_fs_joint` | 344.490 | 18.720 | 7.816 | 92.099 | 11.041 | 0.374 | 0.729 | 103.583 | 0.733 | 11.106 | 0.369 | 0.728 | 1368.558 |
| `exp37_fs_both` | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |
| `exp37_fs_both_b1` | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 | 未同步 |

原始聚合指标：[gen 离散](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/discrete/summary.json)、[gen 连续](outputs_eval/benchmark_v2/exp37_fs_gen/zero_shot/seed_42/continuous/summary.json)、[loc 定位](outputs_loc/benchmark_v2/exp37_fs_loc/zero_shot/seed_42/benchmark_csgo_v2_loc.json)、[joint 定位](outputs_loc/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/benchmark_csgo_v2_loc.json)、[joint 离散](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/discrete/summary.json)、[joint 连续](outputs_eval/benchmark_v2/exp37_fs_joint/zero_shot/seed_42/continuous/summary.json)。这六项的 split／map／数量／seed／checkpoint 与各自 `inference_manifest.json`、`evaluation_plan.json` 一致；生成四图原始指标与清单一致。上述 zero-shot 数值可用于描述未适配模型的跨图表现，不能用于挑选适配配方或推断 few-shot 收益。

## 历史工作与执行优先级

优先：冻结Seen来源与适配配方 → 五组zero-shot → 所选family四图100-shot及retention → 从相同Seen来源独立运行50/20/10-shot及retention。为每个family/shot完成四图后再聚合，不用未完成子集冒充四图macro。

原主线若需补齐：`exp35` joint缺四图×四shot=16次，`exp35_gen/loc`各缺四图×20/10-shot=8次，共32次历史适配；`exp35_gen_de_vertigo` 50-shot本地已有四类推理manifest但缺metric摘要；`exp31_loc/exp32_loc`原表尚缺Unseen zero-shot。`exp36_1`四图四shot已登记完成；这些结果保持原来源和配方，不改名充作exp37结果。历史配置和手动命令已存在，本次不重新创建或启动它们。

## 原实现阶段验收边界（2026-11-03）

当时只执行配置校验、命令计划、CPU单元测试和shell语法检查，未运行train/eval/benchmark子进程，未加载真实模型。后续已同步的 zero-shot 结果见上节；真实400步适配训练及其query、retention评测仍无本机完成证据。配置存在不等于来源已同步或实验已完成。


## 实现验收（2026-11-03）

- `python -B -m unittest tests.test_run_exp37_unseen4 tests.test_fewshot_sampler_epoch -v`：15项CPU测试通过。包含真实sampler的索引检查，以及临时小文件模拟的来源6000／适配400、full后端继承、定位摘要、生成provenance和错误权重／shot／来源拒绝；未加载真实模型、未运行实验子进程。
- `python -B scripts/run_exp37_unseen4.py validate`：80个适配点通过；minimal/full两后端的870个train/infer/metrics/aggregate及zero-shot只读计划检查通过。
- `record.md` 新增六段shell代码均通过 `bash -n`，保留CRLF；新记录表20个family-map行、60个任务macro占位齐全，无伪造指标。新增文档链接、固定矩阵字段和脚本参数一致性已核查。
- `git diff --check`通过；原exp35/36/37 YAML以及旧3maps/map-specific runner未改。训练入口的epoch转发新增开关默认关闭，仅新few-shot开启。
- 原Seen37回归发现工作区另有Stage2 save_steps=1000与旧测试期望2000的差异；本次未修改该并行变更，不将其记为本次few-shot配方变更。
- 2026-11-03 当时三个输出根目录下未创建任何 `exp37_fs*` 实验目录，本机缺少五个Seen Stage2完整来源。这是实现阶段的历史快照；2026-11-04 的来源与zero-shot同步状态以上节为准。
