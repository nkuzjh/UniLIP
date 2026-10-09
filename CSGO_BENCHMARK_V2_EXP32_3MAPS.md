
## exp32_3maps 两轮实验与分阶段筛选

- 本节设计原第一轮 7 组、补充联合对照 J5～J8 共 4 组、第二轮 A0～A3 共 4 组，共 15 组。截至 2026-10-09，已核对第一轮 7 组完成训练及 validation 结果；第二轮尚无已完成的本地评测记录。具体证据见下方实验记录表和实验结果表。
- 固定 Seen-3：`de_ancient, de_dust2, de_nuke`，来自定版 Benchmark v2；训练 15,000 个源样本，验证每地图 500 条。架构沿用 exp32 的共享 LLM LoRA＋两个任务 head LoRA，不切换为 exp31 的冻结 LLM／全量 heads。
- 统一 LoRA `r=32, alpha=64, dropout=0.05`、224 encoder input、短指令、源样本有效 batch 128、最多 50 epoch。单任务每次更新 128 个任务样本；paired joint 每次更新各 128 个定位／生成样本，因此是每任务数据曝光匹配，不是计算量匹配。
- 初始方案为单 GPU、micro batch 4、GAS 32；本批已完成的第一轮七组实际保存的运行参数均为单 GPU、micro batch 64、GAS 2，源样本有效 batch 仍为 128。实际运行布局以各实验 `experiment_plan.json`／Trainer 状态为准；恢复时不改变卡数或 micro batch，跨服务器可改变具体 GPU 编号。
- 所有新配置启用 `csgo_loss_per_microbatch_mean: true`，明确让 Trainer 按 microbatch mean 做 GAS 平均。`joint_original` 表示原联合配方在统一训练实现下的对照，不承诺复刻历史运行数值。
- `scripts/run_exp32_3maps.py` 默认只打印命令，不写配置、不启动子进程；仅加入 `--execute` 才执行一个实验的一个阶段，不会自动提交整轮任务。
- 每次实际训练将最终配置和参数指纹写到 `outputs/csgo_1b/<EXPERIMENT>/resolved_config.yaml` 与 `experiment_plan.json`。推理读取这份配置，不会根据之后修改的 YAML 猜测实际训练设置。

**第一轮配置矩阵**

`S3` 为分段线性调度 `[0,3000,5400,8400] -> [2,5,10,20]`，从 step 0 就开始上升；本批实际最终 step=5850，对应 αloc=11.5，并未到 20。恒定组不保留该 schedule。

| ID | Experiment | 任务 | LLM LoRA LR | Action DiT LoRA LR | Action connector LR | αloc | 默认停止点 |
|---|---|---|---:|---:|---:|---|---:|
| L0 | `exp32_3maps_loc_single` | 定位 | 1e-4 | 1e-4 | 5e-4 | 1 | 2400 |
| G0 | `exp32_3maps_gen_single` | 生成 | 1e-4 | 不更新 | 不更新 | 不参与生成 loss | 2400 |
| J0 | `exp32_3maps_joint_original` | 联合 | 1e-4 | 5e-5 | 1e-4 | S3 | 2400 |
| J1 | `exp32_3maps_joint_headlr` | 联合 | 1e-4 | 1e-4 | 5e-4 | S3 | 2400 |
| J2 | `exp32_3maps_joint_sharedlr` | 联合 | 3e-5 | 1e-4 | 5e-4 | S3 | 2400 |
| J3 | `exp32_3maps_joint_constloc` | 联合 | 1e-4 | 1e-4 | 5e-4 | 恒定 2 | 2400 |
| J4 | `exp32_3maps_joint_sharedlr_constloc` | 联合 | 3e-5 | 1e-4 | 5e-4 | 恒定 2 | 2400 |
| J5 | `exp32_3maps_joint_original_constloc` | 联合 | 1e-4 | 5e-5 | 1e-4 | 恒定 2 | 2400 |
| J6 | `exp32_3maps_joint_constloc1` | 联合 | 1e-4 | 1e-4 | 5e-4 | 恒定 1 | 2400 |
| J7 | `exp32_3maps_joint_constloc4` | 联合 | 1e-4 | 1e-4 | 5e-4 | 恒定 4 | 2400 |
| J8 | `exp32_3maps_joint_constloc8` | 联合 | 1e-4 | 1e-4 | 5e-4 | 恒定 8 | 2400 |

其余定位 norm/projector LR=5e-4，IO/time MLP LR=1e-4，生成 connector/DiT LoRA LR=1e-4；第一轮和补充组全部关闭辅助 loss。J1 对齐定位私有模块，J2/J3/J4 区分共享学习率与主任务权重。J5 只将 J0 的主任务权重固定为 2，保留 J0 全部定位头 LR；J6～J8 仅改变 J3 的恒定 αloc。当前 scheduler 按参数组初始 LR 的比例衰减；LLM LR=3e-5 的末期值约为 3e-6，不是绝对 1e-5。

**第二轮配置矩阵**

四组默认共同使用 J4 的配方，它只是预指定起点，并非已验证最佳。第一轮选定配方后，在第二轮四组的首次 Training 命令中统一设置 `--joint-base exp32_3maps_<joint名称>`；可选 J0～J8 的九个 joint 实验。runner 会复制主任务权重、主 schedule、各 LR 和 LoRA 参数，保留本组辅助设置；四组均从原始 UniLIP-1B 和 pi05 权重独立初始化，不加载第一轮训练后的 checkpoint。续训会沿用已记录的 joint base，禁止中途更换配方。命令中的 `JOINT_BASE` 请在四组启动前统一确定。

| ID | Experiment | aux-loc | perception | αaux 调度 | αperception 调度 | 默认停止点 |
|---|---|---|---|---|---|---:|
| A0 | `exp32_3maps_aux_control` | 关 | 关 | — | — | 3600 |
| A1 | `exp32_3maps_perception` | 关 | 开 | — | `[0,599,600] -> [0,0,0.1]` | 3600 |
| A2 | `exp32_3maps_auxloc` | 开 | 关 | `[0,1800] -> [0,2]` | — | 3600 |
| A3 | `exp32_3maps_both` | 开 | 开 | `[0,1800] -> [0,2]` | `[0,599,600] -> [0,0,0.1]` | 3600 |

aux-loc 沿用两候选 combined EM＋uncertainty、共享定位噪声、`exp_sigma(lambda=5)`；perception 沿用 vision-tower SmoothL1 和 detached GT attention。此轮只改变组件开关，暂不加入弱权重或延迟启用组。若 A0 与第一轮选定 joint 配置完全相同，它是独立的匹配预算对照；不要把不同步数的 J* 与辅助组直接比较。

**最小数据集／全量数据切换**

每份 `csgo_configs/exp32_3maps_*.yaml` 均包含注释。默认：

```yaml
benchmark_v2_manifest: data/csgo_benchmark_v2/benchmark_manifest.json
benchmark_v2_asset_manifest: data/csgo_benchmark_v2/minimal_dataset_report.json
data_dir: data/preprocessed_data
```

启用 asset manifest 时，FPV 和 radar 由最小包报告解析，`data_dir` 是全量后端的备用值，不要求它包含完整资产。同步服务器时需保留 `data/csgo_benchmark_v2` 下 manifest、split JSON、最小报告、images 和 radars 的相对布局。切换全量后端可在 YAML 将 asset manifest 设为 `null`，并设置实际 `data_dir`；更推荐在首次 Training 命令附加：

```bash
# 这是附加参数示例，替换为目标服务器实际路径。
# --asset-mode full --data-dir /path/to/preprocessed_data
```

runner 会将选定后端固化到训练配置，Inference/Metric 默认沿用，不必重复传参。恢复训练需使用相同后端及路径参数，避免指纹冲突。metric 的 `--gt` 会根据最小报告的 `images.target_template` 或全量 `<data_dir>/<map>/imgs` 自动生成；训练、推理、metric 不混用后端。基础权重路径不同可在首次 Training 附加 `--model-path /path/to/UniLIP-1B --pi05-path /path/to/pi05_base`，并在恢复时保持一致。

**Checkpoint 保存、筛选和继续训练**

- 全程固定 `num_train_epochs=50`，采用完整 LR/loss schedule。`training_stop_after_step` 仅在指定绝对 optimizer step 保存并停止，不改变总调度长度；不得用 `--max_steps 2400` 或改为 20 epoch 代替。
- `save_steps=1200, save_total_limit=6, save_only_model=False`。默认保存 `checkpoint-1200/2400/3600/4800`，以及训练结束时的实际最后 checkpoint；非周期停止点也强制保存。checkpoint 包含模型、optimizer/DeepSpeed、scheduler、Trainer 与 RNG 状态，完整恢复使用同 output_dir。
- 预计完整 50 epoch 约 5900 步，具体以目标服务器 Trainer 的 `max_steps`／`global_step` 为准，不将 5900 写成强制 horizon。保存文件较大，最多保留六个 checkpoint；本方案主要节省训练与评测计算，不通过丢弃优化器来省空间。
- 第一轮默认到 2400 步（约 20 epoch）；第二轮默认到 3600 步（约 30 epoch），此时 aux-loc 达到完整权重后已有约 1800 步观察窗口。
- `seen_validation` 是唯一筛选集。每次必须用完整生成采样和相同 seed，查看三图逐图指标与等地图宏平均；单图验证 FID 样本较少，结合 LPIPS/PSNR/Boundary F1 和定位各分量判断，不凭一个波动指标淘汰。保留定位和生成表现互有优势的候选。
- 第一轮优胜者、L0/G0 与 J0 等必要参照按相同步数继续到 3600、4800，必要时完整 50 epoch；第二轮至少让 A0 和各组件代表到 3600，再保留候选到相同的 4800／最终节点。不要求所有 15 组一律跑满。J5～J8 在 2400 初筛后至少复核 3600，必要时续到实际最终节点；已有 J0～J4 在 2400 与 5850 的排序变化，不以 2400 单点结果直接淘汰。
- 离散测试和连续测试仅在配方／checkpoint 通过 validation 选定后报告，不反过来选参数。联合模型两个任务使用同一个 checkpoint。

| 保存节点 | 建议评测 | 是否作淘汰判断 |
|---|---|---|
| 1200，约 10 epoch | 可选：优先定位验证；出现异常再补生成验证 | 只发现明显异常，不因辅助尚未成熟就淘汰 |
| 2400，约 20 epoch | 第一轮及补充组全部有效任务的 validation 推理＋metric | 短程初筛，不单点淘汰 |
| 3600，约 30 epoch | 第二轮四组、J5～J8 及必要参照的 validation | 第二轮主要筛选点；比较预算需匹配 |
| 4800，约 40 epoch | 入围者的 validation | 检查优势是否持续 |
| 实际最后 step，约 5900 | 仅必要候选完成 50 epoch 后的 validation | 先确定 checkpoint，再做 test 报告 |

**Validate configuration / dry run**

```bash
set -euo pipefail
python scripts/run_exp32_3maps.py validate
# 不加 --execute：只打印最终命令，不创建输出目录，不启动训练。
python scripts/run_exp32_3maps.py train --experiment exp32_3maps_joint_sharedlr_constloc
```

**Resume one selected experiment**

```bash
set -euo pipefail
EXPERIMENT=exp32_3maps_joint_sharedlr_constloc
# 完整恢复最新 checkpoint，保持 50 epoch horizon；不要传 resume_ckpt_path 做仅权重加载。
python scripts/run_exp32_3maps.py train --experiment "$EXPERIMENT" --resume --stop-after-step 3600 --cuda-devices 0 --execute
python scripts/run_exp32_3maps.py infer --experiment "$EXPERIMENT" --checkpoint-step 3600 --split validation --cuda-devices 0 --execute
python scripts/run_exp32_3maps.py metrics --experiment "$EXPERIMENT" --checkpoint-step 3600 --split validation --cuda-devices 0 --execute
# 如继续：把停止点改为 4800；只有确定需要跑满时才使用 --stop-after-step 0。
# 不改变同次实验的 seed、GPU 数、micro batch、base、loss、LR 或资产路径。
```

**Report-only test for the validation-selected checkpoint**

```bash
set -euo pipefail
EXPERIMENT=exp32_3maps_joint_sharedlr_constloc
STEP=3600  # 替换为 validation 实际选定的已保存 checkpoint，不必是最后一步。
# 联合模型：定位＋离散生成＋连续生成；单任务模型自动只执行有效任务。
python scripts/run_exp32_3maps.py infer --experiment "$EXPERIMENT" --checkpoint-step "$STEP" --split test --cuda-devices 0 --execute
python scripts/run_exp32_3maps.py metrics --experiment "$EXPERIMENT" --checkpoint-step "$STEP" --split test --cuda-devices 0 --execute
```

定位推理自带指标与 `summary.json`，无需额外定位 metric 子进程；生成 metric 使用 `benchmark_v2_core`，计算正式主表／附录所需指标，不依赖额外 csgosquare 定位器或 aesthetic checkpoint。连续评测保留原 clip、stride、FVD 配置。输出按 experiment、checkpoint、validation/test 和 inference seed 隔离：

```text
outputs/csgo_1b/<EXPERIMENT>/checkpoint-<STEP>/
outputs_loc/benchmark_v2/<EXPERIMENT>/checkpoint-<STEP>/<validation|test>/seed_42/
outputs_eval/benchmark_v2/<EXPERIMENT>/checkpoint-<STEP>/<validation|test>/seed_42/discrete/
outputs_eval/benchmark_v2/<EXPERIMENT>/checkpoint-<STEP>/test/seed_42/continuous/
```

本批实验统一使用新增的 validation 汇总支持、正确传递的生成 RNG、模型加载后重置的采样 seed，以及多圈预测也合法的 yaw 最短角距离。历史已有数字未被重算或覆盖，新实验不能声称与历史推理逐值复现。最初配置验收仅做静态检查、CPU 单元测试和 dry run；现已同步的训练、验证产物及实际运行参数见下表。本次文档更新只核对已有产物。


## exp32_3maps 实验记录表

更新日期：2026-10-09。以下按本地已同步的 `trainer_state.json`、`experiment_plan.json`、checkpoint 模型文件和评测 JSON 核对；“未发现本地记录”不代表远端尚未运行。原第一轮 7 组均有 `global_step=max_steps=5850`、`epoch=50` 及非空最终模型文件；已核对 13 份定位指标和 13 份离散生成汇总。

已完成的原第一轮七组的运行计划和启动参数均记录为单 GPU、microbatch=64、GAS=2、seed=42，源样本有效 batch=128；Trainer 状态记录的 train batch 也为 64。J4 的[训练日志](logs/csgo_1b/exp32_3maps_joint_sharedlr_constloc/train_20261008_030333/train.log)另有对应启动参数证据。L0、G0、J0 的计划记录了从本实验 checkpoint-2400 续训；表中的 5850 是总步数。

| ID | Experiment | 训练 | 推理 | Metric 计算 | validation | checkpoint steps | 训练证据 |
|---|---|---|---|---|---|---|---|
| L0 | `exp32_3maps_loc_single` | ✅ 50 epoch；5850/5850 | ✅ 定位：2400／3600／5850 | ✅ 定位 loc_metrics；无生成任务 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_loc_single/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_loc_single/experiment_plan.json) |
| G0 | `exp32_3maps_gen_single` | ✅ 50 epoch；5850/5850 | ✅ 离散生成：2400／3600／5850 | ✅ 离散生成三图汇总；无定位任务 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_gen_single/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_gen_single/experiment_plan.json) |
| J0 | `exp32_3maps_joint_original` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_original/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_original/experiment_plan.json) |
| J1 | `exp32_3maps_joint_headlr` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_headlr/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_headlr/experiment_plan.json) |
| J2 | `exp32_3maps_joint_sharedlr` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_sharedlr/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_sharedlr/experiment_plan.json) |
| J3 | `exp32_3maps_joint_constloc` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_constloc/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_constloc/experiment_plan.json) |
| J4 | `exp32_3maps_joint_sharedlr_constloc` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、4800、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_sharedlr_constloc/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_sharedlr_constloc/experiment_plan.json) |
| J5 | `exp32_3maps_joint_original_constloc` | 待运行 | 待 validation 推理 | 待指标计算 | 计划 seen_validation；seed=42 | 计划首评 2400、复核 3600；必要时评实际最后节点 | — |
| J6 | `exp32_3maps_joint_constloc1` | 待运行 | 待 validation 推理 | 待指标计算 | 计划 seen_validation；seed=42 | 计划首评 2400、复核 3600；必要时评实际最后节点 | — |
| J7 | `exp32_3maps_joint_constloc4` | 待运行 | 待 validation 推理 | 待指标计算 | 计划 seen_validation；seed=42 | 计划首评 2400、复核 3600；必要时评实际最后节点 | — |
| J8 | `exp32_3maps_joint_constloc8` | 待运行 | 待 validation 推理 | 待指标计算 | 计划 seen_validation；seed=42 | 计划首评 2400、复核 3600；必要时评实际最后节点 | — |
| A0 | `exp32_3maps_aux_control` | 未发现本地训练记录 | 未发现本地结果 | 未发现本地指标 | 待记录 | 未发现本地 checkpoint；计划首评 3600 | — |
| A1 | `exp32_3maps_perception` | 未发现本地训练记录 | 未发现本地结果 | 未发现本地指标 | 待记录 | 未发现本地 checkpoint；计划首评 3600 | — |
| A2 | `exp32_3maps_auxloc` | 未发现本地训练记录 | 未发现本地结果 | 未发现本地指标 | 待记录 | 未发现本地 checkpoint；计划首评 3600 | — |
| A3 | `exp32_3maps_both` | 仅发现计划／配置；无完成证据 | 未发现本地结果 | 未发现本地指标 | 待记录 | 未发现本地 checkpoint；计划首评 3600 | [计划](outputs/csgo_1b/exp32_3maps_both/experiment_plan.json)／[配置](outputs/csgo_1b/exp32_3maps_both/resolved_config.yaml) |

- `checkpoint steps` 列列出本地当前有非空模型文件的节点；这不等于每个节点都已评测。原第一轮七组均有 3600 checkpoint，但当前仅 L0、G0 有 3600 validation 结果；J4 的 4800 checkpoint 也暂无本地评测结果。未列出的保存节点不据此推断从未生成。
- 第一轮的定位与离散生成均在三图各 500 条 `seen_validation` 上完成：定位逐样本文件已核对每图 500 条、共 1,500 个唯一 `(map, file_frame)`；生成汇总逐图 `Common_Count=500`、`Coverage_GT=Coverage_Pred=1`。
- 定位的 13 份 `loc_metrics.json` 已包含逐图指标及 `metrics_macro_map`，但本地未发现对应的独立 `summary.json`；下表直接使用已有宏平均。生成的 13 份 `summary.json` 已存在。
- A3 现有计划使用 J4 配方、单 GPU、microbatch=4、GAS=32、seed=42；缺少 Trainer 状态和 checkpoint，仅登记为配置／计划已存在。A0 尚无独立记录，也未将 J4 的结果自动复制为 A0；若后续确认复用，须注明来源及相同步数、实际配置。
- 尚未发现本轮任何实验的本地 test 定位、test 离散生成或连续生成结果。以下指标均为 validation，不混入历史 exp31 三图测试结果。

## exp32_3maps 实验结果表

表中数值均取源 JSON 的 `metrics_macro_map`，对 `de_ancient`、`de_dust2`、`de_nuke` 做等地图宏平均，保留三位小数；全部使用 inference seed=42。不同 checkpoint 单独成行，按配置矩阵的实验顺序、各实验内步数递增排列。联合模型同一步数的两个任务对应同一 checkpoint。源文件链接保留完整精度和逐图结果；不将不同步数、不同 split 的指标合并比较。

J5～J8 尚未运行，下列定位与生成表暂不填指标；完成同一 checkpoint 的 validation 推理与 metric 后再增行。

### 定位 validation 结果

<!-- | L0 | `exp32_3maps_loc_single` | seen_validation | 3600 | 68.659 | 3.441 | 4.250 | 30.891 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_loc_single/checkpoint-3600/validation/seed_42/loc_metrics.json) | -->

| ID | Experiment | validation | checkpoint steps | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 源文件 |
|---|---|---|---:|---:|---:|---:|---:|---|
| L0 | `exp32_3maps_loc_single` | seen_validation | 2400 | 72.732 | 4.412 | 3.278 | 42.125 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_loc_single/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| L0 | `exp32_3maps_loc_single` | seen_validation | 5850 | 48.459 | 2.878 | 2.818 | 26.235 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_loc_single/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J0 | `exp32_3maps_joint_original` | seen_validation | 2400 | 131.159 | 8.237 | 3.325 | 59.230 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_original/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J0 | `exp32_3maps_joint_original` | seen_validation | 5850 | 82.517 | 4.018 | 2.752 | 34.065 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_original/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J1 | `exp32_3maps_joint_headlr` | seen_validation | 2400 | 128.640 | 6.646 | 3.393 | 58.310 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_headlr/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J1 | `exp32_3maps_joint_headlr` | seen_validation | 5850 | 88.997 | 4.515 | 3.060 | 33.879 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_headlr/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J2 | `exp32_3maps_joint_sharedlr` | seen_validation | 2400 | 149.063 | 7.017 | 4.763 | 62.676 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_sharedlr/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J2 | `exp32_3maps_joint_sharedlr` | seen_validation | 5850 | 89.496 | 4.512 | 2.557 | 37.365 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_sharedlr/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J3 | `exp32_3maps_joint_constloc` | seen_validation | 2400 | 112.408 | 4.857 | 2.752 | 55.544 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J3 | `exp32_3maps_joint_constloc` | seen_validation | 5850 | 67.537 | 3.412 | 2.626 | 35.161 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J4 | `exp32_3maps_joint_sharedlr_constloc` | seen_validation | 2400 | 140.667 | 6.547 | 3.936 | 57.089 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_sharedlr_constloc/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J4 | `exp32_3maps_joint_sharedlr_constloc` | seen_validation | 5850 | 87.428 | 4.485 | 2.918 | 36.186 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_sharedlr_constloc/checkpoint-5850/validation/seed_42/loc_metrics.json) |

### 离散生成 validation 结果

<!-- | G0 | `exp32_3maps_gen_single` | seen_validation | 3600 | 13.932 | 0.419 | 0.636 | 0.505 | 62.949 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_gen_single/checkpoint-3600/validation/seed_42/discrete/summary.json) | -->
| ID | Experiment | validation | checkpoint steps | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 源文件 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| G0 | `exp32_3maps_gen_single` | seen_validation | 2400 | 13.284 | 0.391 | 0.665 | 0.499 | 75.977 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_gen_single/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| G0 | `exp32_3maps_gen_single` | seen_validation | 5850 | 14.402 | 0.418 | 0.613 | 0.520 | 62.252 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_gen_single/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J0 | `exp32_3maps_joint_original` | seen_validation | 2400 | 13.521 | 0.407 | 0.652 | 0.497 | 67.498 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_original/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J0 | `exp32_3maps_joint_original` | seen_validation | 5850 | 14.592 | 0.425 | 0.606 | 0.523 | 61.835 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_original/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J1 | `exp32_3maps_joint_headlr` | seen_validation | 2400 | 13.529 | 0.398 | 0.653 | 0.499 | 69.793 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_headlr/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J1 | `exp32_3maps_joint_headlr` | seen_validation | 5850 | 14.515 | 0.421 | 0.608 | 0.524 | 60.631 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_headlr/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J2 | `exp32_3maps_joint_sharedlr` | seen_validation | 2400 | 13.238 | 0.396 | 0.666 | 0.496 | 66.886 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_sharedlr/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J2 | `exp32_3maps_joint_sharedlr` | seen_validation | 5850 | 14.208 | 0.421 | 0.632 | 0.508 | 64.740 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_sharedlr/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J3 | `exp32_3maps_joint_constloc` | seen_validation | 2400 | 13.267 | 0.389 | 0.659 | 0.501 | 68.925 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J3 | `exp32_3maps_joint_constloc` | seen_validation | 5850 | 14.464 | 0.419 | 0.609 | 0.522 | 65.062 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J4 | `exp32_3maps_joint_sharedlr_constloc` | seen_validation | 2400 | 13.308 | 0.387 | 0.669 | 0.495 | 65.208 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_sharedlr_constloc/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J4 | `exp32_3maps_joint_sharedlr_constloc` | seen_validation | 5850 | 14.111 | 0.413 | 0.630 | 0.512 | 63.516 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_sharedlr_constloc/checkpoint-5850/validation/seed_42/discrete/summary.json) |

### 测试集与第二轮结果记录

| 范围 | split | 结果状态 |
|---|---|---|
| 定位测试 | seen_discrete_test | 未发现本地结果，暂不填写指标 |
| 离散生成测试 | seen_discrete_test | 未发现本地结果，暂不填写指标 |
| 连续生成测试 | seen_continuous | 未发现本地结果，暂不填写 PSNR／SSIM／LPIPS／时序指标／FVD；按方案在 validation 选定 checkpoint 后报告 |
| A0～A3 辅助 loss 消融 | seen_validation | 尚无完成的本地评测结果；训练与配置状态见上方实验记录表 |

以下分析保留为本轮实验启动前的设计依据；本轮已完成实验的实际进度和数值以上述记录及源文件为准。

以下分析为新增 J5～J8 前的历史设计讨论；本节顶部矩阵与记录表为当前执行方案。

## GPT-6 Astra 对当前exp31* exp32*实验结果的分析

我建议继续保持你的核心方向：**先争取“联合训练不差于单任务”，再验证辅助损失能否同时改善两个任务。** 现有证据尚不足以放弃这条路线，但下一轮应把“优化配置不一致”“共享参数干扰”“辅助监督路径不合适”分别验证。

这次补充核对了 exp31 三图消融的已有结果，它们提供了一个重要线索：**perception 值得继续保留，aux-loc 则需要更仔细地检查作用路径。** 以下方案均为讨论建议，没有修改文件或启动实验。

先看现有结果能支持什么判断。

在 exp32 中，联合训练的定位 XY/Yaw 从单任务的 `48.961/26.047` 退化到 `123.194/55.449`；增加辅助损失后恢复到 `63.038/31.671`。这说明辅助损失已经明显缓解定位退化，但还没有达到单任务水平。生成方面，原联合训练在多数重建和连续一致性指标上略有改善，加入辅助后多数指标下降。[Table 1](/home/jiahao/task/UniLIP/csgo_benchmark_v2_table1_seen10.md)

更值得关注的是 exp31 三图实验：

| 冻结 LLM、全量训练 heads | XY↓ | Yaw↓ | 离散 LPIPS↓ | 离散 FID↓ | 连续 LPIPS↓ | FVD↓ |
|---|---:|---:|---:|---:|---:|---:|
| 无辅助：`exp31_1_3maps` | 83.911 | 35.434 | 0.624 | 29.829 | 0.616 | 886.168 |
| 仅 aux-loc：`exp31_2_3maps` | 93.755 | 40.475 | 0.623 | 30.130 | 0.611 | 831.028 |
| 仅 perception：`exp31_3_3maps` | 82.114 | 35.501 | 0.607 | 30.430 | 0.586 | 836.473 |

这里，perception 改善了多数生成指标，定位变化相对较小；aux-loc 改善了连续生成，但定位变差。两者都没有全面占优，FID 也没有改善。[已有三图结果](/home/jiahao/task/UniLIP/csgo_benchmark_v2_experiments_results.md:218)

这些是不同 checkpoint 步数下的历史单次结果，且 exp31 与 exp32 的可训练参数不同，因此不能直接推断 LoRA 设置下的因果关系。但它们足以说明：**目前不应将 exp32_1 的表现解释为两个辅助损失都无效。**

下一轮训练前，我会先处理三类比较口径问题。

| 优先级 | 已核实的问题或风险 | 下一步验证 |
|---|---|---|
| 最高 | 当前代码存在特定条件下梯度累积未正确平均的风险；历史运行是否触发尚未确认 | 固定有效 batch、样本和噪声，比较 GAS=1/16 的裁剪前梯度 |
| 高 | 联合模型的定位头学习率低于定位单任务 | 先对齐定位私有模块学习率，再判断负迁移 |
| 高 | 生成评测的独立随机数生成器没有完整传递；另有资产后端和 yaw 越界计算风险 | 统一评测条件，核实是否影响已有结果 |

GAS 检查不能只比较 Adam 更新后的参数。Adam 对持续的梯度缩放具有一定不敏感性，梯度裁剪也可能掩盖缩放差异，所以应检查累积完成、裁剪之前的梯度。[Adam 原文](https://arxiv.org/abs/1412.6980)

定位头学习率差异则已经明确：

| 参数组 | `exp32_loc` 有效值 | `exp32` |
|---|---:|---:|
| Action DiT LoRA | `1e-4` | `5e-5` |
| Action connector | `5e-4` | `1e-4` |

因此，当前定位差距中可能包含“私有定位模块适应不足”，还不能全部归因于共享 LLM 冲突。配置及分组依据见 [exp32 配置](/home/jiahao/task/UniLIP/csgo_configs/exp32.yaml:69)、[优化器参数分组](/home/jiahao/task/UniLIP/unilip/train/nonmix_trainer.py:1022)。

同时，当前 balanced 模式每个源样本产生定位、生成两个任务，各自在自己的子 batch 内求均值。这里不存在简单的定位 loss 被混合 batch 稀释一半；修改 `task_mix_ratio` 也不会改变该模式实际的 1:1 配对。

**第一轮建议采用 7 组核心实验，把联合训练基线调清楚。**

三张地图仍用 `de_ancient、de_dust2、de_nuke`，训练配置从 exp32 派生。沿用 exp31 三图的数据范围和消融组织方式，同时保留 exp32 的 LoRA 架构。

| 实验 | 设置 | 定位头 LR | 共享 LLM LoRA LR | 定位 loss 权重 | 目的 |
|---|---|---|---:|---|---|
| L0 | 定位单任务 | 原单任务配置 | `1e-4` | `1` | 三图定位参照 |
| G0 | 生成单任务 | — | `1e-4` | — | 三图生成参照 |
| J0 | 原联合配置 | 原联合配置 | `1e-4` | 原调度映射到三图 | 重建联合基线 |
| J1 | J0＋对齐定位头 LR | 与 L0 一致 | `1e-4` | 同 J0 | 排查私有头适应不足 |
| J2 | J1＋降低共享 LR | 与 L0 一致 | `3e-5` | 同 J0 | 降低共享表示更新幅度 |
| J3 | J1＋固定任务权重 | 与 L0 一致 | `1e-4` | 恒定 `2` | 检查递增权重的影响 |
| J4 | J2＋固定任务权重 | 与 L0 一致 | `3e-5` | 恒定 `2` | 检查两项调整的组合效果 |

其中，J1—J4 构成一个小型二维对照：共享学习率高／低，定位权重递增／固定。

我最关注 **J2 和 J4**。它们允许定位头更充分地适应，同时减小共享 LLM 的更新幅度，符合目前的退化模式。但 `αloc=2` 只是候选，不代表梯度已经平衡。若结果需要，再补 `1` 或 `5`；共享 LR 也只在必要时补 `1e-5`，避免直接展开大网格。

这里有几个实施细节需要预先明确：

- 三图原调度可参考现有 `[0,3000,5400,8400] → [2,5,10,20]`。它是**分段线性增加**，不是到节点才跳变；若训练约 5900 步，最终权重约为 11.67，并不会达到 20。
- 只修改 `alpha_loc_loss` 不足以固定权重，还需要处理 schedule 覆盖。
- 只降低 `llm_lora_lr`，保留任务 heads 的独立学习率。
- 增大 `αloc` 与提高定位头 LR 的作用不同：前者还改变共享参数上的任务混合，不能相互替代。
- 如果降低共享 LR 有效，应给予两个单任务模型相应的调参机会，再作最终比较。

若训练早期出现明显梯度尖峰，再单独比较 `warmup_ratio=0.003` 与 `0.02`。暂不把 LoRA rank、dropout、分辨率同时加入首轮搜索。

**第二轮固定最佳联合配置，先拆分辅助损失，再调强度和时间。**

根据已有三图结果，我会优先查看 perception-only 的表现，但完整保留四格对照：

| 实验 | aux-loc | perception | 要回答的问题 |
|---|---|---|---|
| A0 | 关 | 关 | 最佳联合基线 |
| A1 | 关 | 原设置，最高 `0.1` | perception 在共享 LLM LoRA 下是否仍有效 |
| A2 | 原设置，最高 `2.0` | 关 | aux-loc 单独影响哪些任务 |
| A3 | 开 | 开 | 两者是否互补，还是叠加后产生干扰 |

A0 可以复用选定的联合配置，只新增三组。这里复用的是**配置**；各组应采用相同初始化和训练预算。若从训练完成的联合 checkpoint 继续加辅助，则必须增加等预算的无辅助续训对照。

随后根据结果进行有限搜索：

| 现象 | 首选调整 | 候选值 |
|---|---|---|
| aux-loc 恢复定位但伤害生成 | 降低 aux-loc 权重 | `2 → 0.5`，必要时 `0.1` |
| perception 单独伤害生成 | 降低 perception 权重 | `0.1 → 0.02`，必要时 `0.005` |
| 辅助开启后训练明显扰动 | 延迟启用、平滑增加 | 前 20% 关闭，20%—40% 线性增加 |
| 两者单独有效，组合变差 | 降低组合强度 | 组合各自较弱版本 |
| perception 仍无效 | 单独检查注意力加权 | 保持其余设置，关闭 attention weighting |

**降权和延迟最好分开比较。** 如果只测试“更弱＋更晚”的组合，可以判断整套配置是否有效，但无法区分具体是哪项设计带来了收益。

在 50 epoch 预算下，20%—40% 的调度相当于前 10 epoch 关闭辅助，第 10—20 epoch 逐渐增加。此时不能在第 20 epoch 就淘汰辅助方案，至少应再给它一段完整权重下的训练时间，例如观察到第 30 epoch。

另一个原则是：如果辅助梯度本来就很弱，继续降权没有明显依据。最好结合加权后的梯度贡献决定方向。

**目前辅助损失最值得深入的地方，是它究竟在改进生成图像，还是在改变评价器。**

当前 aux-loc 冻结定位私有 head，但共享 LLM 仍可训练，因此共享参数上的辅助梯度包含两部分：

\[
\nabla_\theta L_{\mathrm{aux}}
=
\underbrace{\frac{\partial L}{\partial I}\frac{\partial I}{\partial\theta}}_{\text{经过生成图像}}
+
\underbrace{\left.\frac{\partial L}{\partial\theta}\right|_{I\text{ 固定}}}_{\text{直接更新评价器}}.
\]

这与“定位恢复很多、生成多数变差”的现象相容，但尚不是因果证明。[辅助冻结范围](/home/jiahao/task/UniLIP/unilip/model/language_model/unified_unilip.py:2832)

如果纯配置不能实现目标，我会优先做下面的代码级消融：

| 辅助方式 | 图像反传路径 | 评价器参数直接更新路径 |
|---|---|---|
| 当前实现 | 保留 | 保留 |
| 图像路径约束 | 保留 | 阻断 |
| 评价器路径对照 | 阻断 | 保留 |

“图像路径约束”需要让评价器这次调用不更新自身参数，同时保留对输入图像的导数；整段 `no_grad()` 会切断需要的路径。

此外，当前辅助监督使用带真实图像信息的 noisy latent 的单步重建，并非完整采样图像。因此，训练辅助 loss 下降不一定意味着真实推理时的几何一致性改善。有效候选可以增加一项机制诊断：**用固定、独立的定位模型评估完整采样图像的 pose 一致性**，同时检查该定位器确实依赖 FPV。它作为补充分析，不替代现有 benchmark 指标。

现有反向 `aux_gen_loss` 暂不适合直接开启：辅助提示词仍含真实 pose，新增姿态映射模块默认随机初始化且冻结。需要先处理这些条件与校准问题，才适合用于验证“生成促进定位”。

论文和开源方法方面，我会按解决的问题选择，而不一次叠加多种技术：

| 方法依据 | 对本项目的启示 | 建议顺序 |
|---|---|---|
| [GradNorm，ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | 不同任务的梯度幅度需要平衡，不能只比较 loss 数值 | 静态权重搜索不足时考虑 |
| [PCGrad，NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) | 任务梯度冲突可以通过梯度投影缓解 | 确认共享梯度干扰后使用 |
| [Janus，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Janus_Decoupling_Visual_Encoding_for_Unified_Multimodal_Understanding_and_Generation_CVPR_2025_paper.html) | 理解与生成对表示的需求存在差异，共享方式需要设计 | 支持检查共享范围，不能直接证明本项目的 LLM 冲突 |
| [REPA，ICLR 2025](https://sihyun.me/REPA/) | 高质量表示监督能够帮助生成，但监督位置和目标很重要 | 当前 perception 无效时，再考虑表示层面的改造 |
| [Cross-Task Consistency，CVPR 2020](https://arxiv.org/abs/2006.04096) | 跨任务一致性有方法依据，也已有充分先例 | 用于定位相关工作和收窄创新点 |

尤其要区分：REPA 将干净图像的外部表示对齐到生成模型的中间表示；当前 perception 比较的是重建图像与真实图像的视觉特征。两者不是同一种监督，不能由 REPA 的成功直接推断当前 perception 应该有效。

梯度诊断还应包含两个项目特有因素：

- **5 个有效维度与 27 个填充维度。** 当前 32 维平均 loss 可写成
  \[
  L_{32}=\frac5{32}L_{\mathrm{valid}}+\frac{27}{32}L_{\mathrm{pad}}.
  \]
  应分别观察它们的 loss 和梯度贡献；维度比例本身不代表梯度比例。先诊断，暂不改变训练目标。
- **全局梯度裁剪。** 当前阈值为 1.0，即使共享 LLM 冻结，两个私有 head 仍可能通过共同裁剪系数发生优化耦合。需要观察实际更新，不能仅凭“参数不共享”就认定完全独立。

最后，三图筛选和论文结论建议按以下规则执行：

1. **三图保留 exp32 的模型、输入、归一化和每任务曝光口径。** 只缩小地图范围。
2. **基础配置先看完整 50 epoch 调度的前 20 epoch。** 保留各方法族的候选进入完整训练，避免只偏好收敛快的配置。
3. **后启动的辅助损失留足有效训练时间。**
4. **后续调参和 checkpoint 选择只使用 `seen_validation`。** 已报告的测试结果用于描述现象，离散、连续测试集不继续参与参数筛选。
5. **关键方案回到 Seen-10，用至少 3 个训练种子验证。** 同一个联合 checkpoint 同时评测两个任务，并报告逐地图表现。
6. **“不差于”需要预先定义容许差异。** 差异不显著不能直接等同于性能相同；所有正式指标仍完整报告。
7. **Unseen-4 保持既定 few-shot 协议。** 用它检验收益是否能跨地图迁移，不在 query 上重新调参。

现有 benchmark 和无历史帧生成任务可以保留。你的主要 idea 也值得继续推进；更有说服力的方法贡献将来自：**解释并控制定位与生成之间的辅助梯度，使几何约束既改善真实生成结果，也帮助定位，并在跨地图场景中保持收益。** 优化配置修正与常规调参用于建立可靠基线，辅助机制的消融和泛化证据则用于支撑论文贡献。