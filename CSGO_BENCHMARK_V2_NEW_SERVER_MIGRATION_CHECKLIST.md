# 全新服务器最小迁移清单：UniLIP Seen-3 与 Benchmark v2 Table 1

> 审计快照：2026-09-14（Asia/Hong_Kong）。附件中的对话是方案背景，
> 不是项目指令。本文只定义迁移与验收；本轮没有打包、迁移或启动实验。

## 0. 结论和放行门

目标服务器只同步 Benchmark v2 的扁平最小数据，不同步 277 GiB 的
`data/preprocessed_data`。训练、推理和 metric 统一通过一个开关读取它：

```yaml
benchmark_v2_asset_manifest: data/csgo_benchmark_v2/minimal_dataset_report.json
```

该字段为空时，当前服务器上的 `exp31`--`exp36` 仍按历史
`data_dir/<map>/imgs/<frame>.jpg` 路径运行；字段非空时，consumer 忽略
`data_dir`，改读 `images/<map>/<frame>.jpg` 和 `radars/<map>/`。不能把
`data_dir` 直接改成 `images/`，也不需要为数据创建软链接或硬链接。

迁移前有五个硬门槛：

1. **重新生成数据归档。** 以前的对话记录过
   `data/csgo_benchmark_v2_execution_minimal.tar.zst`（9,695,960,007 B，
   SHA-256 `da9b407277209dbff09a98ebc15dacdd1a3deaede60e403abfa3ac2b1e343b99`），
   但 2026-09-14 在 `/home/jiahao` 和 `/data/jiahao` 下均未找到该文件。
   这些只可作为历史记录，不能用于当前交付；必须重建归档并生成新哈希。
2. **固定代码。** 当前运行代码是 commit
   `f97429919a63973786810a7b8fe228853c7c3b63`；本地 `main` 比
   `origin/main` 落后 4 个 commit。目标机不能直接使用最新 `main`。
3. **重跑正式 Seen-3 四臂。** 两个正在运行的 arm 使用了不同的单卡
   micro-batch，和文档规定的双卡几何不一致，只能作为 exploratory run。
4. **先修正定位 yaw metric。** `eval_csgo_loc.py` 仍使用
   `min(d, period-d)`；预测越界时可能得到负误差。最终定位数字必须在改为
   modulo 最短圆周距离并补测试后重新计算。
5. **外部模型不是现成任务。** 仓库尚无 X-VLA、RDT、OpenVLA-OFT、
   π0.5 或六个生成模型的 CSGO adapter/runner；Task 2 包含实现和验证工作，
   不是仅复制权重即可运行。

## 1. 当前项目快照

### 1.1 数据和代码

- 当前 unpacked 正式最小集已通过：
  `VERIFY TARGET OK: images=102780 (9732747719 bytes), radars=14
  (3103923 bytes), source_access=not_required`。
- `images/`、`radars/` 当前都是普通文件；未发现 symlink 或 link count > 1。
- `minimal_dataset_report.json` 为 `status=verified`、
  `copy_mode=copyfile_not_move`。
- `data/preprocessed_data` 并非空目录，而是指向 277 GiB 全量数据的旧机
  symlink；它不进入迁移集。
- 仓库 worktree 不是干净状态：`record.md`、
  `csgo_benchmark_v2_experiments_results.md` 和调试图有本地变化，本清单也是
  新文件。运行代码本身相对上述 commit 没有已知未提交改动。迁移时应分别
  保存 commit 和必要的进度文档/patch，不能用一次 `git pull` 覆盖现状。

### 1.2 Seen-3 四臂（17:04 快照；运行进度会继续变化）

| Arm | loss | 实际状态 | 正式性判断 |
|---|---|---|---|
| `exp31_3maps` | aux + perception | 未启动，无输出目录 | 待正式运行 |
| `exp31_1_3maps` | joint-only | 约 `4094/5850`；单卡 batch 32、accumulation 4；已有 checkpoint-4000 | exploratory |
| `exp31_2_3maps` | aux-only | 未启动，无输出目录 | 待正式运行 |
| `exp31_3_3maps` | perception-only | 约 `197/5850`；单卡 batch 64、accumulation 2；尚无 checkpoint | exploratory |

四臂都得到 effective batch 128 并不足以形成严格对照：world size、
micro-batch、分布式 sampler 和数值路径也必须一致。默认决策是让现有运行只
保留为探索证据，在新服务器按第 6 节统一重跑四臂；不要把它们和双卡结果
混入同一正式表。若要改成单卡正式协议，也必须先冻结一套相同的
micro-batch/accumulation，并从同一基座重新开始全部四臂。

### 1.3 Table 1 当前可用性

| 行 | 当前证据 | 迁移后的工作 |
|---|---|---|
| `exp31_1` | final step 19500；Seen-10/CrossMap 三任务已有结果 | final 目录可迁移；在 minimal backend 重新推理/评测 |
| `exp31_gen` | final step 19500；Seen/CrossMap generation 已有结果 | 同上，只做生成 |
| `exp31_loc` | final step 19500；Seen localization 已有结果 | 修复 yaw 后重算；CrossMap 不属于 Table 1 最小范围 |
| `exp31` | 只有 checkpoint-8000/10000；曾评测的 checkpoint-6000 已不在输出目录；无 final model | 完成或按冻结预算重跑，不能用 checkpoint-6000 数字冒充 final |
| 所有外部行 | 当前仓库没有 adapter、配置、权重或正式结果 | 按第 7 节开发、smoke、冻结 recipe 后运行 |

现有结果来自旧 source backend。虽然对应图片应相同，provenance 校验会区分
source 与 minimal backend，因此不应直接把旧 inference/metric artifact 当作
新服务器的 minimal-backend 结果。可迁移 final checkpoint，随后重新推理。
如果 Table 1 最终采用三训练 seed 的协议，现有单 seed UniLIP 结果也只够做
回归/smoke，不足以形成最终 `mean ± std`。

## 2. 最小迁移载荷

### 2.1 数据：默认使用正式最小迁移集

| 相对 `data/csgo_benchmark_v2/` 的内容 | 文件数 | 逻辑大小 |
|---|---:|---:|
| `images/` | 102,780 JPG | 9,732,747,719 B |
| `radars/` | 14 PNG | 3,103,923 B |
| `splits/` | 68 JSON | 20,057,770 B |
| `aggregate/` | 9 | 15,729,141 B |
| `calibration/z_calibration.json` | 1 | 7,303 B |
| `calibration/z_extrema_rows.jsonl` | 1 | 1,445,822 B |
| `benchmark_manifest.json` | 1 | 66,312 B |
| `build_report.json` | 1 | 7,638,336 B |
| `checksums.sha256` | 1 | 8,712 B |
| `selected_images.sha256` | 1 | 10,940,220 B |
| `minimal_dataset_report.json` | 1 | 7,886 B |
| **正式最小迁移集** | **102,878** | **9,791,753,144 B（9.119 GiB）** |

它比执行层最小集只多 24,829,314 B（23.679 MiB），却能保留 calibration、
aggregate、构建报告和完整的 `verify-target` 合同，因此迁移默认选择正式集。
当前 ext4 分配量为 10,007,040,000 B（9.320 GiB）。

关键文件当前 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `benchmark_manifest.json` | `4debad27e0d481d31587a537325d6781247934551a7248885e686ed94046ba47` |
| `minimal_dataset_report.json` | `1b7eff2206934f25c03de6aecd445b8094a3940ad875f9cd1803fe953c0185a4` |
| `selected_images.sha256` | `97e9aa5d820373d73f29bacc1a62ceb8755ee8aaa5c48fd656223f6ee2071433` |
| `z_calibration.json` | `67436a888e0f79520bf1ab156009f7384b3a1ade7638b44513f31b2a9b6c14ee` |
| `z_extrema_rows.jsonl` | `54bcc88f23b5e0b84f539751aa8e13e4299565327752c34a9f5d02950b5e6b5c` |
| `build_report.json` | `479496b558049f0e1d08522b4d2e7925ba12640c8c5824778c7952ce85eebbee` |
| `checksums.sha256` | `b36306fa0327083008b7b4edaad8de9fa5da6bf52f142a807769d4590f922df1` |

`verify-target` 还绑定仓库内的 `csgo_configs/benchmark_v2.yaml`；它不计入上面的
数据大小，当前 SHA-256 为
`42460f9a8c5fff976db6289fa5273a01c67f571fe5db223126f6adb2a0390e19`。

### 2.2 Task 1 代码、环境和模型

| 项目 | 固定版本/当前大小 | 是否必需 |
|---|---|---|
| UniLIP 仓库 | commit `f97429919a63973786810a7b8fe228853c7c3b63` | 必需 |
| UniLIP conda 环境 | Python 3.11.14；目录 9,839,212,251 B（9.163 GiB） | 必需；见第 4 节 |
| `UniLIP-1B/` | HF revision `67914afd22139e4de0d1606c0318ebe974ee67b2`；3,578,357,709 B；`model.safetensors` 3,571,791,238 B，SHA `e1f0903c6f7ad9581d3739bd96d19c091309b9bac396971bf3349ac396b00bc8` | 必需，复制完整普通目录 |
| Pi05 的 UniLIP PyTorch 权重 | `config.json` + `model.safetensors` 共 7,233,650,557 B；model SHA `8587d5e8249bae28b2042df424f4a608624ba0285883e3fe650704fff095a5a5` | UniLIP loc 必需 |
| `OpenGVLab/InternVL3-1B` | revision `4415a3b810e636d11dfa86b0e9ba40bb00535aa8`；本地目录 1,876,529,310 B | 离线必需；联网可按 revision 预取 |
| `OpenGVLab/InternVL3-1B-hf` | revision `014c0583a0d4bedf29fbe2dbff4f865eb998e171`；1,892,362,852 B | 同上 |
| `Efficient-Large-Model/Sana_600M_512px_diffusers` | revision `83d7a190bfd1fd070570a793d2dab5c7a3231b9d`；2,367,056,289 B | 同上 |
| `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` | revision `6f7b3f3b289a439a11ef4fb1034989fd4b9a4766`；1,249,046,213 B | 同上 |

`UniLIP-1B/config.json` 会在模型构造阶段引用上述四个 Hugging Face 模型；
只有合并权重文件并不能保证离线初始化成功。联网服务器可以把它们作为目标机
预取项，隔离网服务器必须把固定 revision 的完整 snapshot 一并迁移。

当前 UniLIP 环境的关键版本是：

```text
torch 2.11.0.dev20260124+cu128    torchvision 0.25.0.dev20260124+cu128
transformers 4.57.3               diffusers 0.36.0
deepspeed 0.16.9                  flash_attn 2.8.3
bitsandbytes 0.45.5               accelerate 1.12.0
datasets 3.6.0                    numpy 1.26.4
timm 1.0.24                       tokenizers 0.22.2
```

`requirements.txt` 写的是 `tokenizers==0.22.1`，不能把它视为当前环境的精确
lock。更重要的是，site-packages 中 Transformers 4.57.3 的 Gemma 文件经过
原地替换，分别与
`unilip/openpi_src/models_pytorch/transformers_4573_replace/gemma/` 匹配：

```text
modeling_gemma.py       000659586577cbe6af4f30868aa13c08a6defe7fd680f3de1f2450289ddda535
configuration_gemma.py 26fd0d0f52730ab32b6f337eb73ae9a26cb71ae0915bc49eced1d0e2ed8d0ec6
```

`setup_transformers.py` 指向较旧的 `transformers_replace`，不可盲目运行。

### 2.3 当前 metric 实现的运行依赖

当前离散/连续 metric 脚本会无条件计算额外的 locator、IS、CLIP 和
Aesthetic，即使 Table 1 最后不显示这些列。因此在不改 metric 代码的前提下，
以下也是运行依赖，而不是论文主指标：

| 项目 | 当前版本/大小 |
|---|---:|
| `csgosquare` 代码 | commit `f9c06af2f455890cd306dc68dad6edb21b74d94b`；完整目录 1,409,721,494 B |
| locator checkpoint | 1,066,611,990 B；SHA `9dd6c50fa3207218b59449cf2919351ed0871725fed5f87a75ccd319d6be7101`（已计入上一行目录） |
| FVD 代码 | `third_party/PyTorch-Frechet-Video-Distance` commit `431844ef10417f661dbe47832831ab0558acb340` |
| I3D | 51,235,320 B；SHA `bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2` |
| `aesthetic_model.pth` | 3,714,759 B；SHA `21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0` |
| CLIP base + large HF cache | 2,924,567,327 B |
| torchmetrics AlexNet + Inception cache | 340,037,270 B |

这些文件当前不是 UniLIP git 仓库的可靠 tracked/submodule 内容，必须单独复制或
按固定 revision 预取。不要把 CSGOSquare 的旧 requirements 安装进 UniLIP
环境；当前 evaluator 使用 UniLIP 环境加载其代码和权重。

### 2.4 可选 checkpoint/结果载荷

- 从头跑 Seen-3 不需要任何旧 experiment checkpoint，只需要 `UniLIP-1B`。
- 为避免重跑 Table 1 的已有 UniLIP 行，可复制 `exp31_1`、`exp31_gen`、
  `exp31_loc` 的**顶层 final 文件**；每个目录约 4.29 GiB，并应整体保留
  `model.safetensors`、`config.json`、tokenizer、`mm_projector.bin`、
  `action_heads.bin`、`gen_projector.bin` 和 trainer state。不要只复制一个
  safetensors 后假设所有加载分支都一致。
- 推理用 final bundle 和训练 resume checkpoint 不同。一个完整 DeepSpeed
  resume checkpoint（含 optimizer/scheduler/RNG）当前约 22.6 GB；仅模型文件
  不能无缝续训。
- 若确需续跑 exploratory `exp31_1_3maps`，只在一个 checkpoint 完整落盘且
  训练已暂停/结束后原子快照整个 checkpoint。不要复制正在写入的输出目录。
- 不迁移整个 `outputs`（约 1.3 TiB）、`outputs_eval`（约 52 GiB）或
  `outputs_loc`（约 2.9 GiB）。只保留必要 summary、per-map metric、
  inference manifest 和运行日志；只有在“不重新推理但要重算图像 metric”时
  才复制对应 `gen_imgs`。

### 2.5 已知载荷下限

- 数据 + UniLIP base + Pi05 PyTorch 权重 + 四个核心 HF snapshot：
  27,988,756,074 B（26.067 GiB），不含代码、环境和 metric 模型。
- 再加当前 conda 环境、CSGOSquare 和上述离线 metric cache：约
  42,557,244,495 B（39.635 GiB），仍不含实验 checkpoint 和传输归档副本。
- 单臂同时保留 3 个完整 resume checkpoint 约需 65--70 GiB；四臂并行建议
  为 checkpoint/output 另留至少 300 GiB。顺序运行并在验收后只留 final 可显著
  降低峰值磁盘占用。
- Task 2 的固定最小容量目前不可给出：外部 checkpoint/revision 尚未冻结。
  先完成 smoke 和 license/weight 审核，再汇总真实字节数；不要用猜测值当预算。

## 3. 不迁移的内容

- `data/csgo_benchmark_v2/audit/`（约 17 GiB）、`audit_archive/`、候选
  CSV/JSONL、review export、jump/extrema context、PDF 和可视化中间文件。
- `calibration/*.template.yaml`；只迁移两份已发布 calibration 文件。
- `data/preprocessed_data/` 的 277 GiB 原始数据及其 symlink。
- 仓库中的 `outputs`、`outputs_eval`、`outputs_loc`、`UniLIP-1B`、
  `csgosquare` 等旧机 symlink 本身；迁移所需 target 为普通目录/文件。
- 无关 Hugging Face/Python/pip/W&B cache、调试图片和无关 experiment。
- Table 1 最小阶段不迁移 appendix-only 的 SANA-1.5、LaVida-O、BAGEL、
  NextStep-1；需要附录时另行冻结版本和预算。

## 4. 环境迁移

当前硬件是单张 NVIDIA RTX PRO 6000 Blackwell Server Edition（97,887 MiB，
compute capability 12.0），driver 580.173.02，CUDA runtime 12.8，BF16 可用。
正式 Seen-3 协议要求两张 GPU；目标机应有两张同型号或至少同架构、均能容纳
per-device batch 4 的 BF16 GPU。不同 GPU/driver 可实现可运行迁移，但不能称为
bitwise 复现，必须记录差异。

严格复现当前 Task 1 时，优先使用 `conda-pack` 生成可重定位环境归档，而不是
`rsync` 原 conda 目录。这样能保留已滚动的 PyTorch nightly 和 Gemma 原地补丁。
目标机必须同为 Linux x86_64 且 ABI/driver 兼容；解包后运行 `conda-unpack`，
再以 `--no-deps` 重装当前仓库的 editable package。若目标 GPU 架构不同，至少
重编译/重装 flash-attn 并完成 smoke test。

源机迁移准备项：

```bash
conda env export -n UniLIP --no-builds > environment.yml
conda list -n UniLIP --explicit > conda-explicit.txt
conda run -n UniLIP python -m pip freeze --all > pip-freeze.txt

# conda-pack 当前未安装；安装后再执行，并为产物生成独立 SHA-256。
conda-pack -p /home/jiahao/miniconda3/envs/UniLIP \
  -o unilip-conda-linux-x86_64.tar.gz
```

干净重建是备选方案，但 `requirements.txt`、`pip freeze` 或 conda YAML 中任何
一个都不足以单独复现当前环境。必须保存确切 nightly wheels/本地 wheelhouse，
安装 `torch==2.11.0.dev20260124+cu128`、对应 torchvision、requirements 和
flash-attn 后，再从 `transformers_4573_replace/gemma/` 应用补丁并核对上面的
两个哈希。

所有外部 baseline 使用独立 conda/uv 环境或容器；不要把 OpenVLA-OFT、RDT、
OpenPI/Puffin 等互相冲突的 torch/transformers 依赖装进 UniLIP 环境。最终图片
和 JSON 再交给固定的 UniLIP metric 环境统一评测。

## 5. 源机打包与目标机落位

### 5.1 代码

推荐在目标机 clone 后 checkout 精确 commit：

```bash
PROJECT_ROOT=/home/jiahao/task/UniLIP
git clone https://github.com/nkuzjh/UniLIP.git "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
git checkout --detach f97429919a63973786810a7b8fe228853c7c3b63
test "$(git rev-parse HEAD)" = f97429919a63973786810a7b8fe228853c7c3b63
```

若目标机不能访问 git remote，源机应创建包含该 commit 的 `git bundle`。
`record.md`、实验结果文档和本清单作为单独 provenance 文档/patch 迁移；不要
把 `_debug_dataset_samples.jpg` 的本地变化混入运行 patch。FVD 目录没有可靠的
`.gitmodules` 映射，应单独 checkout 第 2.3 节所列 commit。

### 5.2 保持“除 asset 参数外配置不变”的目录方案

当前代码含相对路径和一个 Pi05 绝对路径。若要求原配置只增加
`benchmark_v2_asset_manifest` 就可运行，目标机必须采用下列普通文件布局并从
项目根目录启动命令：

```text
/home/jiahao/task/UniLIP/
├── UniLIP-1B/                                  # 普通目录，不是旧 symlink
├── data/csgo_benchmark_v2/                     # 正式 flat bundle
├── csgosquare/                                 # 普通目录
├── aesthetic_model.pth
├── loaded_models/5780f6fd48bed6b4f055c5cac089dbee_i3d_torchscript.pt
└── third_party/PyTorch-Frechet-Video-Distance/

/home/jiahao/.cache/openpi/openpi-assets/checkpoints/pi05_base/
├── config.json
└── model.safetensors
```

Hugging Face 模型仍按原 model ID 读取，但正式运行不能让 `main` 在线漂移。
将第 2.2 节四个模型的**完整 cache 目录**（含 `refs/`、`snapshots/`、`blobs/`）
预置到目标用户的 `~/.cache/huggingface/hub/`；当前源机的 `refs/main` 应严格为：

```bash
HF_CACHE=/home/jiahao/.cache/huggingface/hub
test "$(tr -d '\n' < "$HF_CACHE/models--OpenGVLab--InternVL3-1B/refs/main")" = \
  4415a3b810e636d11dfa86b0e9ba40bb00535aa8
test -d "$HF_CACHE/models--OpenGVLab--InternVL3-1B/snapshots/4415a3b810e636d11dfa86b0e9ba40bb00535aa8"
test "$(tr -d '\n' < "$HF_CACHE/models--OpenGVLab--InternVL3-1B-hf/refs/main")" = \
  014c0583a0d4bedf29fbe2dbff4f865eb998e171
test -d "$HF_CACHE/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171"
test "$(tr -d '\n' < "$HF_CACHE/models--Efficient-Large-Model--Sana_600M_512px_diffusers/refs/main")" = \
  83d7a190bfd1fd070570a793d2dab5c7a3231b9d
test -d "$HF_CACHE/models--Efficient-Large-Model--Sana_600M_512px_diffusers/snapshots/83d7a190bfd1fd070570a793d2dab5c7a3231b9d"
test "$(tr -d '\n' < "$HF_CACHE/models--mit-han-lab--dc-ae-f32c32-sana-1.1-diffusers/refs/main")" = \
  6f7b3f3b289a439a11ef4fb1034989fd4b9a4766
test -d "$HF_CACHE/models--mit-han-lab--dc-ae-f32c32-sana-1.1-diffusers/snapshots/6f7b3f3b289a439a11ef4fb1034989fd4b9a4766"
```

先按这些 revision 预取，再在正式训练/推理/metric 中设置
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。这样 YAML 和模型 ID 不变，同时解析到
已验收的 snapshot；若验证命令失败则不得启动正式实验。若不复制标准 cache，
另一种做法是一次性把所有模型引用改成固定 `snapshots/<revision>` 路径，但这不再
属于“除 asset 参数外配置不变”的兼容布局。

这套方案不要求 `/data/home/jiahao/...`，也不为 Benchmark 数据创建链接。

若目标机用户不是 `jiahao` 或必须使用 `/srv/...`，就无法同时保持所有现有 YAML
原样，因为 `pi05_pytorch_weight_path` 被硬编码为 `/home/jiahao/...`，而
`eval_csgo.py` 还从工作目录读取 `UniLIP-1B`。此时必须做一次系统化的本地 YAML
overlay/路径配置改造；完成该一次性端口化后，日常 source/minimal 数据切换仍只
改 asset 参数。不能声称仅改 asset 参数就能解决模型绝对路径。

### 5.3 重建正式数据归档

以下是待执行模板，不代表归档已经存在：

```bash
set -euo pipefail
SOURCE_REPO=/home/jiahao/task/UniLIP
TRANSFER_ROOT=/path/to/transfer
ARCHIVE_NAME=csgo_benchmark_v2_formal_minimal.tar.zst
test -d "$SOURCE_REPO/data/csgo_benchmark_v2/images"
mkdir -p "$TRANSFER_ROOT"
test ! -e "$TRANSFER_ROOT/$ARCHIVE_NAME"
test ! -e "$TRANSFER_ROOT/$ARCHIVE_NAME.sha256"
cd "$SOURCE_REPO"
conda run -n UniLIP python scripts/materialize_csgo_benchmark_v2.py verify-target

tar --zstd --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 --numeric-owner \
  -cf "$TRANSFER_ROOT/$ARCHIVE_NAME" -C "$SOURCE_REPO" \
  data/csgo_benchmark_v2/benchmark_manifest.json \
  data/csgo_benchmark_v2/build_report.json \
  data/csgo_benchmark_v2/checksums.sha256 \
  data/csgo_benchmark_v2/selected_images.sha256 \
  data/csgo_benchmark_v2/minimal_dataset_report.json \
  data/csgo_benchmark_v2/aggregate \
  data/csgo_benchmark_v2/calibration/z_calibration.json \
  data/csgo_benchmark_v2/calibration/z_extrema_rows.jsonl \
  data/csgo_benchmark_v2/splits \
  data/csgo_benchmark_v2/images \
  data/csgo_benchmark_v2/radars

cd "$TRANSFER_ROOT"
sha256sum "$ARCHIVE_NAME" | tee "$ARCHIVE_NAME.sha256"
sha256sum -c "$ARCHIVE_NAME.sha256"
stat -c '%n %s bytes' "$ARCHIVE_NAME" "$ARCHIVE_NAME.sha256"
```

目标机先验证归档哈希，再从项目根目录解压；以下模板拒绝覆盖已有数据目录：

```bash
PROJECT_ROOT=/home/jiahao/task/UniLIP
TRANSFER_ROOT=/path/to/received
ARCHIVE_NAME=csgo_benchmark_v2_formal_minimal.tar.zst
cd "$TRANSFER_ROOT"
sha256sum -c "$ARCHIVE_NAME.sha256"
test ! -e "$PROJECT_ROOT/data/csgo_benchmark_v2"
tar --zstd -xf "$ARCHIVE_NAME" -C "$PROJECT_ROOT" --no-same-owner

cd "$PROJECT_ROOT"
/path/to/UniLIP-env/bin/python \
  scripts/materialize_csgo_benchmark_v2.py verify-target
find data/csgo_benchmark_v2/images data/csgo_benchmark_v2/radars \
  -type l -print -quit | (! read -r _)
find data/csgo_benchmark_v2/images data/csgo_benchmark_v2/radars \
  -type f -links +1 -print -quit | (! read -r _)
```

### 5.4 目标机环境和代码 smoke

```bash
cd /home/jiahao/task/UniLIP
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/path/to/UniLIP-env/bin/python -m pip install -e . --no-deps
/path/to/UniLIP-env/bin/python -m pip check
/path/to/UniLIP-env/bin/python - <<'PY'
import torch, transformers, diffusers, deepspeed, flash_attn, unilip
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(transformers.__version__, diffusers.__version__, deepspeed.__version__)
print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())
PY

/path/to/UniLIP-env/bin/python -m unittest -q \
  tests.test_benchmark_v2_asset_plumbing \
  tests.test_csgo_benchmark_v2_runtime \
  tests.test_benchmark_csgo_v2_protocol \
  tests.test_aggregate_csgo_benchmark_v2_metrics
```

当前 UniLIP 环境没有 pytest；以上四个模块使用标准库 `unittest`，当前源机实测
44 tests 通过，因此最小迁移无需仅为这组预检额外安装 pytest。

随后对 Seen-3 的一个 train batch、一个 loc batch、一个 discrete gen batch 和
一个 16-frame continuous clip 做 smoke；核对输入文件名、shape、输出尺寸、
显存、seed 和 asset provenance 后才启动正式训练。

## 6. Task 1：UniLIP Seen-3 ablation

### 6.1 冻结协议

| Arm | aux-loc | loc perception |
|---|---:|---:|
| `exp31_3maps` | on | on |
| `exp31_1_3maps` | off | off |
| `exp31_2_3maps` | on | off |
| `exp31_3_3maps` | off | on |

- 地图顺序：`de_ancient, de_dust2, de_nuke`。
- 每臂：15,000 train；6,000 localization/discrete test；20 clips/map，连续
  共 3,840 frames。无 CrossMap、few-shot 或 Seen-10 retention。
- 四臂分别从同一 `UniLIP-1B` 初始化；ViT/LLM frozen，full heads，balanced
  joint training。训练 seed 必须相同并显式记录。
- 正式 launch geometry：2 GPUs × per-device batch 4 × accumulation 16 =
  effective global batch 128，50 epochs。项目协议预计 5,900 optimizer steps；
  以统一正式运行实际生成的 trainer state 为验收事实，任何偏差都要解释。
- 只用 `seen_validation` 选择 checkpoint/超参；两个 test split 只在 recipe
  冻结后运行一次。推理 seed 为 42，结果为 equal-map macro。

### 6.2 训练模板

从仓库根目录运行；四臂只替换 `ARM` 和未占用的 `PORT`，其余 CLI 相同：

```bash
set -euo pipefail
cd /home/jiahao/task/UniLIP
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ASSET_MANIFEST=data/csgo_benchmark_v2/minimal_dataset_report.json
ARM=exp31_3maps
PORT=29568

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port="$PORT" \
  train_csgo.py --csgo_config "csgo_configs/$ARM.yaml" \
  --deepspeed deepspeed_scripts/zero0.json \
  --model_name_or_path UniLIP-1B --unilip_factor 10.6 \
  --mllm_hf_path OpenGVLab/InternVL3-1B-hf \
  --version internvl --data_type mix --csgo_image_folder data/preprocessed_data \
  --benchmark_v2_asset_manifest "$ASSET_MANIFEST" \
  --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True \
  --output_dir "outputs/csgo_1b/$ARM" --num_train_epochs 50 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 --eval_strategy no --save_strategy steps \
  --save_steps 1000 --save_total_limit 3 --learning_rate 1e-4 \
  --weight_decay 0. --warmup_ratio 0.003 \
  --lr_scheduler_type cosine_with_min_lr --model_max_length 1024 \
  --logging_steps 1 --tf32 True --gradient_checkpointing True \
  --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 \
  --n_und_query 0 --report_to wandb --fix_dit False \
  --fix_connect False --fix_llm True --seed 42
```

`data/preprocessed_data` 在这里仅是历史 CLI 占位；asset manifest 非空时 loader
不访问它。启动日志必须显示 `benchmark_v2_asset_manifest` 已激活，不能靠自动
检测。保存 commit、YAML、完整 CLI、环境 hash、world size、实际 batch/step。

### 6.3 推理和 metric

每个 final checkpoint 依次运行：

```bash
cd /home/jiahao/task/UniLIP
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM=exp31_3maps
MAPS=(de_ancient de_dust2 de_nuke)
ASSET=data/csgo_benchmark_v2/minimal_dataset_report.json
CKPT="outputs/csgo_1b/$ARM/model.safetensors"

python eval_csgo_loc.py --csgo_config "csgo_configs/test/${ARM}_loc.yaml" \
  --output_dir "outputs_loc/benchmark_v2/$ARM/seen3" --ckpt_path "$CKPT" \
  --seed 42 --benchmark_v2_split seen_discrete_test \
  --benchmark_v2_maps "${MAPS[@]}" --benchmark_v2_asset_manifest "$ASSET"

python eval_csgo.py --csgo_config "csgo_configs/test/${ARM}_gen.yaml" \
  --output_dir "outputs_eval/benchmark_v2/$ARM/seen3/discrete" \
  --ckpt_path "$CKPT" --seed 42 --benchmark_v2_split seen_discrete_test \
  --benchmark_v2_maps "${MAPS[@]}" --benchmark_v2_asset_manifest "$ASSET"

python eval_csgo.py --csgo_config "csgo_configs/test/${ARM}_gen_conti.yaml" \
  --output_dir "outputs_eval/benchmark_v2/$ARM/seen3/continuous" \
  --ckpt_path "$CKPT" --seed 42 --benchmark_v2_split seen_continuous \
  --benchmark_v2_maps "${MAPS[@]}" --benchmark_v2_asset_manifest "$ASSET"
```

直接运行 metric 时，GT 必须是 flat 路径，并同时传 protocol/asset manifest：

```bash
cd /home/jiahao/task/UniLIP
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM=exp31_3maps
MAP=de_ancient
ASSET=data/csgo_benchmark_v2/minimal_dataset_report.json

python benchmark_csgo_v1.py \
  --gt "data/csgo_benchmark_v2/images/$MAP" \
  --pred "outputs_eval/benchmark_v2/$ARM/seen3/discrete/gen_imgs/$MAP" \
  --map_name "$MAP" --data_dir data/preprocessed_data \
  --benchmark_v2_manifest data/csgo_benchmark_v2/benchmark_manifest.json \
  --benchmark_v2_split seen_discrete_test \
  --benchmark_v2_asset_manifest "$ASSET" --batch_size 1 --device cuda \
  --external_loc_repo_root csgosquare \
  --external_loc_config_path configs_reg_newdata/exp5_2.yaml \
  --external_loc_checkpoint_path \
    checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth

python benchmark_csgo_v1_conti.py \
  --gt "data/csgo_benchmark_v2/images/$MAP" \
  --pred "outputs_eval/benchmark_v2/$ARM/seen3/continuous/gen_imgs/$MAP" \
  --map_name "$MAP" --data_dir data/preprocessed_data \
  --benchmark_v2_manifest data/csgo_benchmark_v2/benchmark_manifest.json \
  --benchmark_v2_split seen_continuous \
  --benchmark_v2_asset_manifest "$ASSET" --batch_size 1 --device cuda \
  --frame_diff_threshold 2 --min_track_len 4 \
  --clip_length 16 --clip_stride 16 --fvd_size 224 \
  --external_loc_repo_root csgosquare \
  --external_loc_config_path configs_reg_newdata/exp5_2.yaml \
  --external_loc_checkpoint_path \
    checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth
```

以上两条 metric 命令按 `MAPS` 顺序各运行三次。它们会把 per-map JSON 写到各自
`gen_imgs` 的上级输出目录；全部成功后执行 equal-map macro：

```bash
cd /home/jiahao/task/UniLIP
ARM=exp31_3maps
MAPS=(de_ancient de_dust2 de_nuke)

python scripts/aggregate_csgo_benchmark_v2_metrics.py maps \
  --manifest data/csgo_benchmark_v2/benchmark_manifest.json \
  --split seen_discrete_test \
  --input_root "outputs_eval/benchmark_v2/$ARM/seen3/discrete" \
  --kind discrete --maps "${MAPS[@]}" \
  --output "outputs_eval/benchmark_v2/$ARM/seen3/discrete/summary_equal_map.json"

python scripts/aggregate_csgo_benchmark_v2_metrics.py maps \
  --manifest data/csgo_benchmark_v2/benchmark_manifest.json \
  --split seen_continuous \
  --input_root "outputs_eval/benchmark_v2/$ARM/seen3/continuous" \
  --kind continuous --maps "${MAPS[@]}" \
  --output "outputs_eval/benchmark_v2/$ARM/seen3/continuous/summary_equal_map.json"
```

不要使用旧 `eval_csgo_v1*.py` wrapper：当前 wrapper 没有把 asset manifest
可靠地传到 metric 参数。

验收条件：每臂 loc 6,000 rows、discrete 6,000 张、continuous 3,840 张；
三地图各 20 个完整 64-frame clip；coverage=1；无 extra/missing file；
inference manifest、per-map JSON 和 summary 的 manifest/asset/checkpoint/seed
完全一致。

## 7. Task 2：Table 1 外部模型

### 7.1 冻结的主表集合

附件中后续出现的候选不覆盖用户明确确认的最终集合。Table 1 出版结构可保持
两个 panel；执行上拆成 localization、discrete generation、continuous
generation 三条独立流水线。

| Panel | 分组 | 固定行 |
|---|---|---|
| Localization | Independent VLA | X-VLA、RDT、OpenVLA-OFT-CSGO |
|  | Shared action prior | π0.5-CSGO† |
|  | Ours | `exp31_loc`、`exp31_1`、`exp31` |
| Generation | Conditional specialist | OmniGen-CSGO、ControlAR-CSGO |
|  | Generic T2I | Lumina-Image-2.0-CSGO |
|  | Unified multimodal | Show-o2-1.5B-CSGO、Puffin-CSGO、Janus-Pro-CSGO 1B |
|  | Ours | `exp31_gen`、`exp31_1`、`exp31` |

`†` 必须披露 π0.5 与 UniLIP 使用共享 action prior；它不是完全独立预训练的
baseline。外部行均应描述为“在相同 Seen-10 上适配的开源模型”，不能写成
原模型 zero-shot 支持 CSGO。

### 7.2 共用协议

- Seen-10：50,000 train（5,000/map）、5,000 validation、20,000 discrete
  test、200 个 64-frame continuous clips（12,800 frames）。Table 1 最小范围
  不含 CrossMap-4；后者属于另表/附录。
- localization 输入为当前 FPV + radar/map + 固定 instruction/map name；state
  为 zero/masked，不能注入 GT pose。输出 normalized absolute
  `[x,y,z,pitch,yaw]`，horizon=1。
- generation 输入为 radar/map + normalized 5DoF；输出一张 448×448 RGB
  FPV。连续任务仍逐帧独立条件生成，不使用 GT 目标帧、历史/未来 GT、depth 或
  额外 geometry。
- pose 不能只转成自然语言。生成 adapter 至少采用 Fourier feature + MLP +
  pose tokens、FiLM 或等价的显式数值条件，并对所有生成模型统一语义和顺序。
- raw 图像方向、crop 语义和 radar orientation 固定；允许模型使用其官方 native
  input resolution，但必须记录。所有输出统一到 448×448 后评测，不能把
  UniLIP 的 `img_size=224` 错当成所有模型的输出尺寸。
- 训练 seed 固定为 0/1/2，inference seed 固定为 42；先用 seed 0 在 validation
  做最多 4 个 recipe，冻结后才以三 seed 运行 test，并报告 mean ± std。
- 公平预算以相同 Seen-10 sample presentations、50 epochs 和目标 effective
  global batch 128 为主；记录每个实现实际 optimizer steps、world size、
  micro-batch 和 accumulation。只有 runner 语义确认相同时才强制写死 19,500
  steps，不能把不同 dataloader 的名义 step 直接等同。
- checkpoint/hyperparameter 只用 validation；test 只运行冻结 recipe。每张地图
  单独计算，再做 equal-map macro。禁止 incomplete/debug coverage。

主指标：

```text
Localization: XY_Dist↓, Z_Dist↓, Pitch_Dist↓, Yaw_Dist↓
Discrete:     PSNR↑, SSIM↑, LPIPS↓, Boundary_F1↑, FID↓
Continuous:   PSNR↑, SSIM↑, LPIPS↓, TWE↓, TDE↓, FVD↓
```

Coverage、Common_Count、Track_Count、Seq_Frame_Count、fvd_clip_count 是完整性
验收字段，不参与质量排名。所有行另报告
`Total loaded params / Task-active params / CSGO-trainable params`，并在附录
记录 peak VRAM、GPU-hours、latency/throughput、预训练权重和许可证。

### 7.3 必须实现的统一 adapter 合同

当前下列文件/命令尚不存在；先实现，不能直接粘贴一个“train_external”模板就
宣称完成：

1. 一个 manifest-driven dataset adapter，只从 asset report 解析 FPV/radar，
   严格按 split row 和 map order 取样。
2. VLA 双图输入 adapter、zero/masked state 和重新初始化的 5D absolute
   horizon-1 head；记录新增/冻结/可训练参数，禁止 GT leakage。
3. 生成模型的 radar encoder/composite 与显式 numeric pose conditioner；输出
   `gen_imgs/<map>/<file_frame>.jpg`，不允许改 sample ID。
4. 统一 inference manifest，至少记录 model/repo commit、weights SHA、adapter
   commit、split/maps、sample count、train/inference seed、checkpoint、protocol
   manifest hash 和 asset report/selected-images hash。
5. localization 输出先只写预测和 sample ID；evaluator 再从 manifest join GT，
   生成与当前 metric 兼容的 `loc_results.json`。模型推理过程不得看到 GT 字段。
6. schema/coverage validator 和统一 train/infer/eval runner；外部环境只写标准
   图片/JSON，最终全部在同一个 UniLIP metric 环境计算。

### 7.4 外部仓库和独立环境

当前仓库未 vendoring 下列代码/权重。每项在目标机按官方来源 clone/download，
smoke 后冻结 commit/tag、checkpoint SHA-256、license 和环境 lock：

| 行 | 官方代码 | 当前 CSGO 状态 |
|---|---|---|
| X-VLA | [2toinf/X-VLA](https://github.com/2toinf/X-VLA) | adapter/权重未落地 |
| RDT | [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) | 同上 |
| OpenVLA-OFT | [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | 同上 |
| π0.5 | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | baseline 未落地；UniLIP 内部仅有转换后的 PyTorch action 权重 |
| OmniGen | [VectorSpaceLab/OmniGen](https://github.com/VectorSpaceLab/OmniGen) | adapter/权重未落地 |
| ControlAR | [hustvl/ControlAR](https://github.com/hustvl/ControlAR) | 同上 |
| Lumina-Image 2.0 | [Alpha-VLLM/Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0) | 同上 |
| Show-o2 | [showlab/show-o](https://github.com/showlab/show-o) | 同上 |
| Puffin | [KangLiao929/Puffin](https://github.com/KangLiao929/Puffin) | 同上 |
| Janus-Pro | [deepseek-ai/Janus](https://github.com/deepseek-ai/Janus) | 同上 |

原始 π0.5/OpenPI baseline 需要完整官方 checkpoint/runtime；UniLIP 所需的
7.23 GB PyTorch `model.safetensors` 不能代替 OpenPI 的 TensorStore/JAX 资产。
当前本机完整 `pi05_base` cache 约 19,675,372,488 B，可作为来源候选，但仍要
按官方 loader、commit 和文件完整性重新验收。

### 7.5 执行顺序

1. 实现公共 schema、dataset adapter、provenance 和 metric bridge，并用一个
   dummy backend 做 full-coverage dry run。
2. 每个模型做单 batch shape/memory smoke，再做 seed 0 的 200--500-step
   convergence smoke；未通过的模型不进入正式矩阵。
3. 每模型最多 4 个 validation recipe；冻结一套 trainability、optimizer、
   conditioning 和 checkpoint 选择规则。
4. 正式运行 seeds 0/1/2，先 train + validation，再一次性跑对应 test：VLA 只跑
   localization；生成模型跑 discrete 和 continuous。
5. 统一 metric、equal-map aggregate、参数/资源统计和 provenance 审计后，才填
   Table 1。原生不支持的任务填 `N/A`，不得填 0 或伪造转换结果。

## 8. 最终验收清单

### 数据与代码

- [ ] 新 formal tar.zst 已重建；源/目标归档 SHA-256 一致，压缩后实际字节数已记录。
- [ ] 解压后 102,878 个 formal 文件和 9,791,753,144 B 逻辑内容一致。
- [ ] `verify-target` 通过；images/radars 无 symlink、hardlink alias、缺失或 extra。
- [ ] 目标代码严格是 commit `f974299...` 加经审核的迁移 patch；没有误用较新的 main。
- [ ] 所有运行的 protocol manifest、asset report、selected checksum、calibration
  hash 与本清单一致。

### 环境与模型

- [ ] UniLIP 环境 import、`pip check`、Gemma 补丁哈希和四项 runtime test 通过。
- [ ] `UniLIP-1B`、Pi05、四个核心 HF 模型、CSGOSquare、FVD/I3D/Aesthetic/
  metric cache 均能在离线 smoke 中加载，或联网预取策略已实际验证。
- [ ] 使用第 5.2 节兼容布局时，旧实验只新增 asset 参数即可运行；使用其他路径时，
  已明确记录一次性 overlay，不再声称只改 asset 参数。
- [ ] 两张训练 GPU、driver/runtime、BF16、显存和跨卡通信 smoke 通过。

### Task 1

- [ ] 四臂从同一 base、相同 seed 和 2×4×16 几何独立启动；exploratory 单卡结果未混入。
- [ ] 每臂实际 final step、checkpoint、trainer state、完整日志和资源用量已保存。
- [ ] 每臂 loc/discrete/continuous 分别完整覆盖 6,000/6,000/3,840；三地图
  equal-map summary 和 inference provenance 通过 strict aggregator。
- [ ] yaw modulo 修复和越界单测已通过，定位 metric 已在修复后重算。

### Task 2

- [ ] 精确 Table 1 模型集合、三 seed、训练预算和 validation recipe 已冻结。
- [ ] 每个外部 repo/weight/license/env/adapter 有 commit、SHA 和参数量记录；环境互相隔离。
- [ ] VLA 无 GT state 泄漏；生成模型使用显式 numeric pose，不读取目标/历史/未来帧。
- [ ] VLA 全覆盖 20,000 localization；生成模型全覆盖 20,000 discrete 和
  12,800 continuous；文件名、clip 顺序、输出 448×448 均通过 schema 检查。
- [ ] 所有最终数字来自相同 metric 代码、相同 flat data identity、equal-map macro；
  coverage 字段完整但不参与排名。
- [ ] `exp31_loc`/`exp31_gen`/`exp31_1` 已按同一正式协议重跑，`exp31` 已完成；
  三 seed 不足时明确标为未完成，不能填正式 mean ± std。

完成以上项目后，最小交付物应是：数据归档及其 SHA、精确代码 commit/patch、
环境归档或 wheel lock、模型 checksum 清单、几何/协议冻结文件、Task 1 四臂的
checkpoint/manifest/metrics，以及 Task 2 每个正式行的 adapter、checkpoint、
inference manifest 和 per-map/aggregate JSON。
