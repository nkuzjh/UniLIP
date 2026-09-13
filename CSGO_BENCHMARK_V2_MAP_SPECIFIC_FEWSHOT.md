# UniLIP-CSGO Benchmark v2 Map-specific Few-shot Design

本文档定义 Benchmark v2 的 map-specific CrossMap-4 few-shot 实验。这里
的 map-specific 指每个模型只适配一张目标地图，而不是用一个模型同时
适配四张 CrossMap 地图。本文档记录 exp35/exp36 以及新增的 exp36_1
实验协议，不改变 exp31-exp34 的训练、推理、评测或已有记录。

## 1. 研究目的

exp33/exp34 使用一个模型同时读取四张 CrossMap 地图的 support（用于
适配的少量样本）。exp35/exp36 将同样的适配预算拆成四个独立模型，
每个模型只读取一张地图的 support。这样可以回答两个问题：

1. 在相同 Seen-10 初始化和相同优化步数下，单地图适配是否比四地图
   联合适配具有更好的目标地图质量；
2. 这种提升是否值得额外的模型数量、训练次数、存储和推理成本。

`exp36_1` 是 `exp31_1` Seen-10 joint full-head 权重到 CrossMap-4 的
map-specific few-shot 适配。它只保留 joint 路由，不建立
`exp36_1_gen_<map>` 或 `exp36_1_loc_<map>` 训练实验；每个适配后的
joint checkpoint 同时评测目标地图的定位、离散生成、连续生成以及
Seen-10 retention。此前出现的 “CrossMap-10” 是笔误，本文统一使用
“CrossMap-4 的 100/50/20/10-shot”。

这里的 `support` 是 few-shot 训练样本，`query` 是只用于最终评测的
测试样本，`Seen-retention` 是适配后回到 Seen-10 地图测试以检查遗忘。

## 2. 实验矩阵

CrossMap-4 固定为：

```text
cs_office, de_golden, de_palacio, de_vertigo
```

每个 `<map>` 的训练 route 如下；`exp36_1` 只建立 joint 模型：

| 方法族 | 联合任务 | generation-only | localization-only |
| --- | --- | --- | --- |
| exp35 full-head | `exp35_<map>` | `exp35_gen_<map>` | `exp35_loc_<map>` |
| exp36 LoRA | `exp36_<map>` | `exp36_gen_<map>` | `exp36_loc_<map>` |
| exp36_1 full-head | `exp36_1_<map>` | 不建立 | 不建立 |

模型名完整展开为：

```text
exp35_cs_office       exp35_gen_cs_office       exp35_loc_cs_office
exp35_de_golden       exp35_gen_de_golden       exp35_loc_de_golden
exp35_de_palacio      exp35_gen_de_palacio      exp35_loc_de_palacio
exp35_de_vertigo      exp35_gen_de_vertigo      exp35_loc_de_vertigo

exp36_cs_office       exp36_gen_cs_office       exp36_loc_cs_office
exp36_de_golden       exp36_gen_de_golden       exp36_loc_de_golden
exp36_de_palacio      exp36_gen_de_palacio      exp36_loc_de_palacio
exp36_de_vertigo      exp36_gen_de_vertigo      exp36_loc_de_vertigo

exp36_1_cs_office
exp36_1_de_golden
exp36_1_de_palacio
exp36_1_de_vertigo
```

初始化关系必须保持如下：

| 新模型 | 直接初始化 checkpoint | 不允许使用 |
| --- | --- | --- |
| `exp35_<map>` | Seen-10 `exp31` | `exp33` 或其他已适配 checkpoint |
| `exp35_gen_<map>` | Seen-10 `exp31_gen` | `exp33_gen` 或其他已适配 checkpoint |
| `exp35_loc_<map>` | Seen-10 `exp31_loc` | `exp33_loc` 或其他已适配 checkpoint |
| `exp36_<map>` | Seen-10 `exp32` | `exp34` 或其他已适配 checkpoint |
| `exp36_gen_<map>` | Seen-10 `exp32_gen` | `exp34_gen` 或其他已适配 checkpoint |
| `exp36_loc_<map>` | Seen-10 `exp32_loc` | `exp34_loc` 或其他已适配 checkpoint |
| `exp36_1_<map>` | Seen-10 `exp31_1` | `exp35`、`exp36` 或其他已适配 checkpoint |

`finetune_init_ckpt_path` 表示只加载模型权重并重新建立 optimizer、
scheduler 和 global step，不是继续恢复原训练的 Trainer 状态。

## 3. 固定协议

每个 map-specific 模型只使用目标地图的
`crossmap_support` rows。训练配置中的 `train_maps`、`val_maps` 和
`test_maps` 也只列出目标地图，避免数据集层面意外混入其他 CrossMap。

固定项如下：

| 项目 | 协议 |
| --- | --- |
| manifest | `data/csgo_benchmark_v2/benchmark_manifest.json` |
| support split | `crossmap_support` |
| support seed | `0`，正式默认值 |
| shot counts | `100`，并支持 `50/20/10` |
| optimization budget | `MAX_STEPS=400` |
| inference seed | `42` |
| CrossMap query | 只评测该模型对应的目标地图 |
| Seen retention | 评测全部 Seen-10 地图 |
| early stopping/tuning | 不得查看 CrossMap query 或 continuous 结果后调参 |

每一个 map、task 和 shot count 都必须从对应的 Seen-10 checkpoint
独立开始。50-shot、20-shot 和 10-shot 不能从 100-shot 的适配模型
继续训练；四张地图之间也不能互相加载适配后的 checkpoint。对
`exp36_1`，这意味着四张地图乘四种 shot 共 16 个独立 joint 训练点，
每个点都直接从 `outputs/csgo_1b/exp31_1/model.safetensors` 初始化。

`SHOTS` 是每张地图的 support 样本数，而不是四张地图合计样本数。默认
`SHOTS=100` 时，每个 map-specific 模型只看 100 行 support；四个模型
合计看 400 行，但每个模型仍然独立优化。

## 4. 方法和配置契约

### exp35 full-head

exp35 继承 exp33 的模型设计、loss、数据增强、prompt 和后处理。联合
任务保留 exp33 的 full-head 训练标志；gen-only 和 loc-only 分别使用
exp31_gen、exp31_loc 的 Seen-10 初始化以及对应单任务开关。

### exp36 LoRA

exp36 继承 exp34 的 LoRA 设计、loss、数据增强、prompt 和后处理。联合
任务保留 exp34 的共享 language-model LoRA、generation-head LoRA 和
localization-head LoRA。gen-only 和 loc-only 使用 exp32_gen、exp32_loc
的对应 LoRA gating。

### exp36_1 full-head joint-only

`exp36_1_<map>` 严格继承 `exp31_1` 的可学习模块、数据增强、prompt 和
后处理：`is_lora=False`，LLM 和 ViT frozen，connect、DiT、generation
head 和 localization head 可训练。`aux_loc_loss`、`loc_perception_loss`
以及其他 aux/perception 变体均关闭；CrossMap 适配时定位主损失固定为
`alpha_loc_loss=20`。这只是从 `exp31_1` Seen-10 权重开始的
map-specific adaptation，不引入新的 split 或其他模型设计。

`exp36_1` 的直接初始化路径固定为：

```text
outputs/csgo_1b/exp31_1/model.safetensors
```

每个 shot 都重新建立 optimizer、scheduler 和 global step，输出路径为：

```text
outputs/csgo_1b/exp36_1_<map>/shot_<N>/seed_0/
```

### 训练 YAML 命名

每个模型的训练配置使用精确模型名：

```text
csgo_configs/exp35_<map>.yaml
csgo_configs/exp35_gen_<map>.yaml
csgo_configs/exp35_loc_<map>.yaml
csgo_configs/exp36_<map>.yaml
csgo_configs/exp36_gen_<map>.yaml
csgo_configs/exp36_loc_<map>.yaml
csgo_configs/exp36_1_<map>.yaml
```

现有 exp35/exp36 配置已实现并完成契约检查：训练配置将
`finetune_init_ckpt_path` 指向上表的直接 Seen-10 checkpoint，并把
Benchmark v2 的 map arrays 限制为一个目标地图；24 个训练 YAML 和 48 个
推理 YAML 的命名、继承关系、目标地图、shot/step、batch 与输出路径均已按
本节约束核对。正式运行前仍需将实际 Seen-10 checkpoint 路径与服务器环境
确认一致。`exp36_1` 新增 4 个训练 YAML 和 12 个 joint 推理 YAML；
配置存在不代表对应训练或评测已经完成。

### 推理 YAML 命名

对联合模型 `NAME=exp35_<map>`、`exp36_<map>` 或 `exp36_1_<map>`，推理配置为：

```text
csgo_configs/test/NAME_gen.yaml
csgo_configs/test/NAME_gen_conti.yaml
csgo_configs/test/NAME_loc.yaml
```

对 gen-only 模型 `NAME=exp35_gen_<map>` 或 `exp36_gen_<map>`，推理配置
为：

```text
csgo_configs/test/NAME_gen.yaml
csgo_configs/test/NAME_gen_conti.yaml
```

对 loc-only 模型 `NAME=exp35_loc_<map>` 或 `exp36_loc_<map>`，推理配置
为：

```text
csgo_configs/test/NAME_loc.yaml
```

所有 map-specific 的定位推理配置（联合模型的 `NAME_loc.yaml` 和
loc-only 模型的 `NAME_loc.yaml`）显式设置
`benchmark_v2_allow_map_subset_summary: True`。这是定位摘要的专用 gate：
map-specific 评测只输入一个目标地图时，允许
`build_localization_summary` 接受该 CrossMap 子集；其默认值仍为 `False`，
完整协议评测继续要求全部协议地图。exp31-exp34 的配置和严格行为不变。

例如 `exp35_cs_office` 的三个联合推理配置是
`exp35_cs_office_gen.yaml`、`exp35_cs_office_gen_conti.yaml` 和
`exp35_cs_office_loc.yaml`；`exp35_gen_cs_office` 的两个生成配置是
`exp35_gen_cs_office_gen.yaml` 和
`exp35_gen_cs_office_gen_conti.yaml`。`exp36_1_cs_office` 的三个 joint
推理配置是 `exp36_1_cs_office_gen.yaml`、
`exp36_1_cs_office_gen_conti.yaml` 和 `exp36_1_cs_office_loc.yaml`；
其他三张地图遵循同一命名规则。

## 5. Batch 和训练命令约定

联合训练使用：

```text
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=32
```

这与 exp33/exp34 的联合适配命令一致，`exp36_1` 也使用同样的
`4/4/32` 设置。gen-only 和 loc-only 使用：

```text
BATCH_SIZE=SHOTS       # 100、50、20 或 10
per_device_eval_batch_size=128
gradient_accumulation_steps=1
```

原因是单任务数据集只包含一张地图，task-homogeneous sampler（同一
batch 只放同一任务的采样器）会丢弃不完整 batch。若固定使用 batch 128，
20-shot 或 10-shot support 甚至可能没有一个完整 batch；令训练 batch
等于 `SHOTS` 可确保每个 shot 设置至少产生一个完整 batch。

所有适配命令都应使用 `--max_steps 400`，并将输出写入：

```text
outputs/csgo_1b/<NAME>/shot_${SHOTS}/seed_0/
```

详细、可直接执行的逐实验命令位于
[`record.md`](record.md) 的 `# csgo_benchmark_v2` 章节末尾。

`exp36_1` 的正式顺序命令使用 shot 级屏障：

```bash
for SHOTS in 100 50 20 10; do
  python scripts/run_csgo_benchmark_v2_map_specific.py schedule \
    --families exp36_1 \
    --routes joint \
    --shots "$SHOTS" \
    --cuda-device 0
done
```

一次 `schedule` 内，四张地图按实时空闲显存尽量并行；同一地图模型内部的
训练、三类推理、metric 和汇总保持串行。外层循环只有在当前 shot 的四图
pipeline 与 family macro 全部完成后才进入下一个 shot，因此不会提前启动
50/20/10-shot。

## 6. 推理和评测

### CrossMap query

每个模型的生成推理必须分别运行：

- `crossmap_query_test`，只传入它对应的目标地图；
- `crossmap_continuous`，只传入它对应的目标地图。

联合模型（包括 `exp36_1_<map>`）和 gen-only 模型执行离散及连续生成。
loc-only 模型只执行
`crossmap_query_test` 定位推理；定位没有单独的连续生成阶段。

每个模型的生成输出放在：

```text
outputs_eval/benchmark_v2/<NAME>/shot_${SHOTS}/seed_0/discrete/
outputs_eval/benchmark_v2/<NAME>/shot_${SHOTS}/seed_0/continuous/
```

定位输出放在：

```text
outputs_loc/benchmark_v2/<NAME>/shot_${SHOTS}/seed_0/
```

### Seen retention

同一个 map-specific checkpoint 还要在全部 Seen-10 地图上运行：

```text
seen_discrete_test
seen_continuous
```

Seen-retention 的输出目录使用 `seen_retention_discrete`、
`seen_retention_continuous` 和 `seen_retention` 子目录。生成的 Seen
retention 结果可以用现有 `maps` 子命令汇总，因为这里是同一个 checkpoint
在多个地图上的结果。定位 JSON 由 `eval_csgo_loc.py` 直接产生。

### 单地图生成 metric

CrossMap query 的 generation metric 必须对目标地图直接运行
`benchmark_csgo_v1.py` 或 `benchmark_csgo_v1_conti.py`。不能对单个
map-specific 模型调用现有 `maps` 汇总，因为 `maps` 的语义是一个
checkpoint 覆盖 manifest 规定的全部协议地图。

### Map-models family macro

四个 map-specific checkpoint 的 CrossMap family macro 使用新增的
`map-models` 子命令。它的含义是：从四个不同 checkpoint 各取一个目标
地图结果，再对四张地图等权平均。因此它与 `maps`（一个 checkpoint 的
多地图平均）和 `seeds`（同一方法的 support selection 平均）不同。

生成宏平均只覆盖生成路线，定位宏平均只覆盖定位路线。命令中的 `{map}`
由 CLI 展开为四张 CrossMap 地图：

```bash
set -euo pipefail
V2_MANIFEST=data/csgo_benchmark_v2/benchmark_manifest.json
SHOTS=100
SUPPORT_SEED=0
GENERATION_FAMILIES=(exp35 exp35_gen exp36 exp36_gen exp36_1)
LOCALIZATION_FAMILIES=(exp35 exp35_loc exp36 exp36_loc exp36_1)

for FAMILY in "${GENERATION_FAMILIES[@]}"; do
  for KIND in discrete continuous; do
    if [[ "$KIND" == discrete ]]; then
      SUBDIR=discrete
      RESULT_NAME='benchmark_csgo_v2_{map}.json'
      SPLIT=crossmap_query_test
    else
      SUBDIR=continuous
      RESULT_NAME='benchmark_csgo_v2_conti_{map}.json'
      SPLIT=crossmap_continuous
    fi
    python scripts/aggregate_csgo_benchmark_v2_metrics.py map-models \
      --manifest "$V2_MANIFEST" \
      --split "$SPLIT" \
      --kind "$KIND" \
      --input_pattern "outputs_eval/benchmark_v2/${FAMILY}_{map}/shot_${SHOTS}/seed_${SUPPORT_SEED}/${SUBDIR}/${RESULT_NAME}" \
      --output "outputs_eval/benchmark_v2/${FAMILY}/shot_${SHOTS}/seed_${SUPPORT_SEED}/map_models/benchmark_v2_${KIND}_crossmap.json"
  done
done

for FAMILY in "${LOCALIZATION_FAMILIES[@]}"; do
  python scripts/aggregate_csgo_benchmark_v2_metrics.py map-models \
    --manifest "$V2_MANIFEST" \
    --split crossmap_query_test \
    --kind localization \
    --input_pattern "outputs_loc/benchmark_v2/${FAMILY}_{map}/shot_${SHOTS}/seed_${SUPPORT_SEED}/benchmark_csgo_v2_loc.json" \
    --output "outputs_loc/benchmark_v2/${FAMILY}/shot_${SHOTS}/seed_${SUPPORT_SEED}/map_models/benchmark_v2_localization_crossmap_query_test.json"
done
```

这里的 `map-models` 输出只用于四个专门模型的等地图宏平均。论文中还
必须同时保留四张地图的原始结果和 `map -> checkpoint` 映射，不能只报告
宏平均而隐藏模型专门化。

## 7. 公平性和成本报告

exp35/exp36 的每个 task route 都需要四个模型，约为 exp33/exp34 一个
四地图模型的 4 倍 CrossMap adaptation runs；如果联合、gen-only、
loc-only 三条 route 全部运行，则是 12 个 map-specific 模型。`exp36_1`
另外为每种 shot 运行 4 个 joint map-specific 模型，因此四种 shot 合计
16 个独立适配点，而不是 4 个共享四地图模型。因此比较必须同时报告：

1. 每张目标地图的质量和四地图等权 macro；
2. 训练 GPU hours、optimizer steps、support 样本数和 wall-clock time；
3. trainable parameter 数、checkpoint 存储量和推理时间；
4. Seen-retention 的质量变化，用于判断地图专门化是否造成遗忘。

map-specific 结果在目标地图上更好，只能说明专门化模型的目标地图适配
更强；只有在质量、成本和 Seen-retention 一起比较后，才能判断它是否
优于 exp33/exp34 的单模型联合适配方案。

## 8. 可复现检查清单

训练前逐个确认：

- 模型名、地图名和训练 YAML 名字完全一致；
- `finetune_init_ckpt_path` 是对应 exp31* 或 exp32* 的 Seen-10 权重；
- `exp36_1` 的 parent 是 `outputs/csgo_1b/exp31_1/model.safetensors`，
  `is_lora=False`，LLM/ViT frozen，connect/DiT 与 gen/loc heads 可训练；
- `benchmark_v2_split=crossmap_support`、`support_seed=0`、`SHOTS=100`；
- 目标地图是唯一的 train/validation/test map；
- `MAX_STEPS=400`，联合 batch 为 `4/4/32`，单任务 batch 为 `SHOTS/128/1`；
- `exp36_1` 只使用 joint 路由；不存在 `exp36_1_gen_<map>` 或
  `exp36_1_loc_<map>` 训练名；
- `exp36_1` 的 aux-loc/perception loss 关闭，CrossMap 定位主损失固定为
  `alpha_loc_loss=20`；
- 推理使用 `seed=42`，CrossMap query 只包含目标地图；
- Seen-retention 使用全部 Seen-10 地图；
- metric 输入的 manifest、split、support metadata 和 checkpoint provenance
  与推理输出一致；
- `map-models` 汇总前，四张地图的结果文件各自完整且来自对应的
  map-specific checkpoint。
