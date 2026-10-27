
## exp32_3maps 两轮实验与分阶段筛选

- 原两轮设计含第一轮 7 组、补充联合对照 J5～J10 共 6 组、第二轮 A0～A3 共 4 组，以及 2026-10-14 新配置的 J8 辅助 A1_J8～A3_J8 共 3 组；新增独立的 J8 Stage 2 四组 S0_J8～S3_J8 后为历史 24 个独立运行名；2026-10-23 再设计九组消融，共 33 个独立运行名。其中 S0 是无辅助的等更新预算归因对照，S1～S3 是用户要求的三个辅助主组。原第一轮与 J5～J10 保留 2026-10-13 的核对快照；**2026-10-22 已补充 S0_J8～S3_J8：四组均完成 local 1800，local 1200／1800 的定位和离散生成 validation 结果已同步**；同日补充 **A1_J8 已完成 50 epoch、5900 steps，2400／5900 两任务 validation 结果齐全**。**2026-10-26 九组新消融本地快照：P1／P2／U2 已完成 local1800，并齐备 local1200／1800 的定位与离散生成 validation、local1800 的定位／离散／连续 test；其他六组未取得完整本地评测。** **2026-10-27 更新：U1／P3 已完成 local1800，local1200／1800 定位＋离散生成 validation 与 local1800 定位／离散／连续 test 均已同步；九组中现有五组完整。连同 L0／G0／J8@5850、S0_J8～S3_J8@local1800，共 12 组 test，见 [test结果总表](#test-overall-20261027)。** 现有配置编号为 S0（control）、S1（perception）、S2（aux-loc）、S3（both），未发现独立 S4_J8 配置或产物，不将 S2 结果改标为 S4。具体证据与分析见下方记录表及 [Stage 2 结果分析](#stage2-results-20261022)。
- **2026-10-27 补充 U5_S2_J8**：在上述九组之外新增独立运行名 `exp32_3maps_U5_S2_J8`，计划总数由 33 增至 34；仅将 U2_S2_J8 的 aux-loc ramp 终值 1 降至 0.5，当前仅配置、未启动训练或评测。配置、进度及空结果表见[独立 U5 补充](#u5-aux05-20261027)。
- **2026-10-28 补充 P5_S1_J8**：再新增独立运行名 `exp32_3maps_P5_S1_J8`，当前计划总数由 34 增至 35（原九组＋U5＋P5 共 11 个新增独立运行名）；仅将 P3_S1_J8 的 perception ramp 终值 0.05 降至 0.025，当前仅配置、未启动训练或评测。配置、进度及空结果表见[独立 P5 补充](#p5-perception025-20261028)。
- 固定 Seen-3：`de_ancient, de_dust2, de_nuke`，来自定版 Benchmark v2；训练 15,000 个源样本，验证每地图 500 条。架构沿用 exp32 的共享 LLM LoRA＋两个任务 head LoRA，不切换为 exp31 的冻结 LLM／全量 heads。
- 原两轮统一 LoRA `r=32, alpha=64, dropout=0.05`、224 encoder input、短指令、源样本有效 batch 128、最多 50 epoch；新 Stage 2 保留相同模型与名义有效 batch，但显式使用 1800 optimizer steps 的新调度。满批设计口径下，单任务每次更新 128 个任务样本，paired joint 每次更新各 128 个定位／生成样本，目标是匹配每任务数据曝光而非计算量；本次 Stage 2 的实际布局存在 epoch 尾批例外，曝光并非严格匹配，见结果分析。
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
| J9 | `exp32_3maps_joint_constloc12` | 联合 | 1e-4 | 1e-4 | 5e-4 | 恒定 12 | 2400 |
| J10 | `exp32_3maps_joint_constloc16` | 联合 | 1e-4 | 1e-4 | 5e-4 | 恒定 16 | 2400 |

其余定位 norm/projector LR=5e-4，IO/time MLP LR=1e-4，生成 connector/DiT LoRA LR=1e-4；第一轮和补充组全部关闭辅助 loss。J1 对齐定位私有模块，J2/J3/J4 区分共享学习率与主任务权重。J5 只将 J0 的主任务权重固定为 2，保留 J0 全部定位头 LR；J6～J10 仅改变 J3 的恒定 αloc。当前 scheduler 按参数组初始 LR 的比例衰减；LLM LR=3e-5 的末期值约为 3e-6，不是绝对 1e-5。

**第二轮配置矩阵**

原 A0～A3 四组保持 J4 默认配方、原 YAML、命令与输出目录；2026-10-14 新增三个独立的 J8 辅助运行名 A1_J8～A3_J8。纳入 J9/J10 后，2026-10-13 的同口径评测建议 J8 作为新辅助组的固定主配方，J9 保留为定位优先候选，依据见下方[最新结果分析](#j9-j10-results-20261013)。新组可省略 `--joint-base` 或显式设置 `--joint-base exp32_3maps_joint_constloc8`；runner 固定 J8 配方并拒绝切换其他 base，复制主任务权重、主 schedule、各 LR 与 LoRA 参数，同时保留本组辅助设置。A1_J8～A3_J8 分别从原始 UniLIP-1B 与 pi05 权重独立初始化，不加载 J8 训练后的 checkpoint；续训只恢复自身 checkpoint 且沿用已保存配方。A0 不新增 J8 版本，是否以 J8 作为无辅助对照须先核实初始化、seed、主配方、数据布局、有效 batch、步数和评测口径。

| ID | Experiment | JOINT_BASE | aux-loc | perception | αaux 调度 | αperception 调度 | 默认停止点 |
|---|---|---|---|---|---|---|---:|
| A0 | `exp32_3maps_aux_control` | J4（原默认） | 关 | 关 | — | — | 3600 |
| A1 | `exp32_3maps_perception` | J4（原默认） | 关 | 开 | — | `[0,599,600] -> [0,0,0.1]` | 3600 |
| A2 | `exp32_3maps_auxloc` | J4（原默认） | 开 | 关 | `[0,1800] -> [0,2]` | — | 3600 |
| A3 | `exp32_3maps_both` | J4（原默认） | 开 | 开 | `[0,1800] -> [0,2]` | `[0,599,600] -> [0,0,0.1]` | 3600 |
| A1_J8 | `exp32_3maps_perception_j8` | J8（固定） | 关 | 开 | — | `[0,599,600] -> [0,0,0.1]` | 3600 |
| A2_J8 | `exp32_3maps_auxloc_j8` | J8（固定） | 开 | 关 | `[0,1800] -> [0,2]` | — | 3600 |
| A3_J8 | `exp32_3maps_both_j8` | J8（固定） | 开 | 开 | `[0,1800] -> [0,2]` | `[0,599,600] -> [0,0,0.1]` | 3600 |

aux-loc 沿用两候选 combined EM＋uncertainty、共享定位噪声、`exp_sigma(lambda=5)`；perception 沿用 vision-tower SmoothL1 和 detached GT attention。此轮只改变组件开关，暂不加入弱权重或延迟启用组。J8 主配方固定 αloc=8，LLM LoRA／Action LoRA LR=1e-4，Action connector／norm／projector LR=5e-4，IO LR=1e-4；新组三者的辅助开关及调度逐项沿用旧 A1～A3。默认单卡 microbatch=64、GAS=2、源样本有效 batch=128，50 epoch horizon，首停 3600，save_steps=1200、save_total_limit=6。J8 作为新组三者的无辅助参照须核实原始权重初始化、seed=42、配方、数据布局、有效 batch、评测步数及同一推理设置。当前本地 J8 只有 2400／5850 的 validation 指标；3600 主筛选须取得并补评 J8@3600，同步不到则只记录辅助组结果，不拿 J8@5850 跨步数比较。

**独立的 J8 Stage 2 矩阵（2026-10-17 设计；2026-10-22 已同步结果）**

详细配方、归因与验证口径见 [J8 两阶段辅助微调设计](CSGO_BENCHMARK_V2_EXP32_3MAPS_STAGE2.md)，逐组操作见 [record.md](record.md)。四组均仅加载 J8 `checkpoint-5850/model.safetensors` 的模型权重；新 optimizer、scheduler、Trainer/RNG 状态从 Stage 2 local step 0 开始，不继承 J8 的 optimizer 或 50 epoch 调度。与从原始 UniLIP-1B／pi05 权重独立训练的旧 A1_J8～A3_J8 不同，新组不使用、也不接受 `--joint-base`。J8@5850 是初始化参照，S0_J8 才是同节点辅助归因对照：设计要求它与三个主组匹配更新数、主任务曝光和 LR；本次实际匹配了更新数与 LR，但 batch 布局／尾批使曝光略不等，辅助计算也使 GPU 时间不匹配；两阶段总成本须计入共享的 J8 Stage 1 成本。

| ID | Experiment | perception | aux-loc | Stage 2 local 调度 | 默认停止／验证节点 |
|---|---|---|---|---|---|
| S0_J8 | `exp32_3maps_stage2_control_j8` | 关 | 关 | 无辅助；αloc=8 | 首停 1200；600／1200／按需 1800 |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | 开 | 关 | `[0,120] -> [0,0.1]`；αloc=8 | 首停 1200；600／1200／按需 1800 |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | 关 | 开 | `[0,120] -> [0,2]`；αloc=8 | 首停 1200；600／1200／按需 1800 |
| S3_J8 | `exp32_3maps_stage2_both_j8` | 开 | 开 | 两项均按 local `[0,120]` 线性启用；αloc=8 | 首停 1200；600／1200／按需 1800 |

Stage 2 设计固定 `max_steps=1800`、seed=42，默认单卡 microbatch=64／GAS=2（本次实际布局见记录表），所有 J8 参数组峰值 LR ×0.2（普通组 2e-5；Action connector/norm/projector 1e-4），新建 AdamW 与 cosine min LR 调度（warmup 60 步、min LR rate 0.1）；每 300 local steps 保存，最多 6 份完整 checkpoint。默认 1200 保存后退出，需手动推理与评测，再决定是否以完整 Stage 2 checkpoint 恢复到 1800。验证在同一 checkpoint 上做 seen_validation 三图各 500 条的定位和离散生成，比较四组 1200 同节点结果；600 供观察，1800 仅作匹配预算的补充。`--checkpoint-step` 使用 Stage 2 local 600／1200／1800，绝不填写累计 6450／7050／7650 或 J8 的 5850。

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

**原两轮 Checkpoint 保存、筛选和继续训练（不适用于独立 Stage 2）**

- 原两轮全程固定 `num_train_epochs=50`，采用完整 LR/loss schedule。`training_stop_after_step` 仅在指定绝对 optimizer step 保存并停止，不改变总调度长度；不得用 `--max_steps 2400` 或改为 20 epoch 代替。
- `save_steps=1200, save_total_limit=6, save_only_model=False`。默认保存 `checkpoint-1200/2400/3600/4800`，以及训练结束时的实际最后 checkpoint；非周期停止点也强制保存。checkpoint 包含模型、optimizer/DeepSpeed、scheduler、Trainer 与 RNG 状态，完整恢复使用同 output_dir。
- 原两轮预计完整 50 epoch 约 5900 步，具体以目标服务器 Trainer 的 `max_steps`／`global_step` 为准，不将 5900 写成强制 horizon。保存文件较大，最多保留六个 checkpoint；本方案主要节省训练与评测计算，不通过丢弃优化器来省空间。
- 第一轮默认到 2400 步（约 20 epoch）；第二轮默认到 3600 步（约 30 epoch），此时 aux-loc 达到完整权重后已有约 1800 步观察窗口。
- `seen_validation` 是唯一筛选集。每次必须用完整生成采样和相同 seed，查看三图逐图指标与等地图宏平均；单图验证 FID 样本较少，结合 LPIPS/PSNR/Boundary F1 和定位各分量判断，不凭一个波动指标淘汰。保留定位和生成表现互有优势的候选。
- 第一轮优胜者、L0/G0 与 J0 等必要参照按相同步数继续到 3600、4800，必要时完整 50 epoch；第二轮至少让 A0 和各组件代表到 3600，再保留候选到相同的 4800／最终节点。不要求原两轮 20 组一律跑满。J5～J10 已完成到 5850，并有 2400／5850 的 validation 结果；若补评 3600，须先确认对应 checkpoint 已同步，且不以 2400 单点结果直接淘汰。
- 离散测试和连续测试仅在配方／checkpoint 通过 validation 选定后报告，不反过来选参数。联合模型两个任务使用同一个 checkpoint。

| 保存节点 | 建议评测 | 是否作淘汰判断 |
|---|---|---|
| 1200，约 10 epoch | 可选：优先定位验证；出现异常再补生成验证 | 只发现明显异常，不因辅助尚未成熟就淘汰 |
| 2400，约 20 epoch | 第一轮及补充组全部有效任务的 validation 推理＋metric | 短程初筛，不单点淘汰 |
| 3600，约 30 epoch | 原第二轮四组、新 J8 辅助三组、J8@3600 及必要参照的 validation | 第二轮主要筛选点；比较预算需匹配 |
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

定位推理自带指标并写入 `loc_metrics.json`，正式定位摘要文件名为 `benchmark_csgo_v2_loc.json`，无需额外定位 metric 子进程；生成 metric 使用 `benchmark_v2_core`，计算正式主表／附录所需指标，不依赖额外 csgosquare 定位器或 aesthetic checkpoint。连续评测保留原 clip、stride、FVD 配置。输出按 experiment、checkpoint、validation/test 和 inference seed 隔离：

```text
outputs/csgo_1b/<EXPERIMENT>/checkpoint-<STEP>/
outputs_loc/benchmark_v2/<EXPERIMENT>/checkpoint-<STEP>/<validation|test>/seed_42/
outputs_eval/benchmark_v2/<EXPERIMENT>/checkpoint-<STEP>/<validation|test>/seed_42/discrete/
outputs_eval/benchmark_v2/<EXPERIMENT>/checkpoint-<STEP>/test/seed_42/continuous/
```

本批实验统一使用新增的 validation 汇总支持、正确传递的生成 RNG、模型加载后重置的采样 seed，以及多圈预测也合法的 yaw 最短角距离。历史已有数字未被重算或覆盖，新实验不能声称与历史推理逐值复现。最初配置验收仅做静态检查、CPU 单元测试和 dry run；现已同步的训练、验证产物及实际运行参数见下表。本次文档更新只核对已有产物。


## exp32_3maps 实验记录表

更新日期：2026-10-22（更新 Stage 2 四组及 A1_J8，其余行保留原快照）。2026-10-13 快照已核对原第一轮及 J5～J10 的 25 份定位指标和 25 份离散生成汇总（定位含 L0@3600，生成含 G0@3600）；当时 J5～J10 的远端训练完成由用户确认，部分 checkpoint／元数据仍在同步。以下旧组的同步描述仍指该日期，不代表本次重新扫描的状态。

本次新增核对 Stage 2 **8 份定位指标＋8 份离散生成汇总**，四组均为 `global_step=max_steps=1800`，每组 local 1200／1800 的两任务结果均齐全。local600 没有找到本地评测产物，保留空值；不将训练完成等同于每个保存点都已评测。四组均指向 J8@5850 的相同模型来源，但实际采用双卡：S0 为 microbatch=64、GAS=1；S1/S2 为 microbatch=8、GAS=8；S3 为 microbatch=4、GAS=16。名义有效 batch 均为 128，实际 epoch 分别为 15.3846／15.2559／15.2560，不能直接写为严格相同主任务曝光。具体归因限制见本次分析。

A1_J8 本次另核对 **2 份定位指标＋2 份离散生成汇总**。训练状态为 `global_step=max_steps=5900`、`epoch=50`；运行计划记录双 GPU、microbatch=4、GAS=16、seed=42，名义有效 batch=128，并记录从自身 checkpoint-3600 恢复。当前本地模型节点为 2400、5900 及根目录，已评测节点为 2400／5900；没有发现 3600 的本地评测产物。此布局每 epoch 对应 118 次更新，50 epoch 为 5900 steps，因此不能将最终节点误记成 J8 的 5850。按已保存配置，A1_J8 从原始权重独立训练并使用 J8 主配方，是 perception 消融；它与从 J8 权重开始的 S1_J8 两阶段分支不同。

已完成的原第一轮七组的运行计划和启动参数均记录为单 GPU、microbatch=64、GAS=2、seed=42，源样本有效 batch=128；Trainer 状态记录的 train batch 也为 64。J4 的[训练日志](logs/csgo_1b/exp32_3maps_joint_sharedlr_constloc/train_20261008_030333/train.log)另有对应启动参数证据。L0、G0、J0 的计划记录了从本实验 checkpoint-2400 续训；表中的 5850 是总步数。

J5～J9 已同步的运行计划记录单 GPU、microbatch=64、GAS=2、seed=42、50 epoch，源样本有效 batch=128；根目录 Trainer 状态的 train batch 为 64。J9 计划另记录 save_steps=1200、save_total_limit=6；J10 运行计划与训练状态尚待同步核对。

| ID | Experiment | 训练 | 推理 | Metric 计算 | validation | checkpoint steps | 训练证据 |
|---|---|---|---|---|---|---|---|
| L0 | `exp32_3maps_loc_single` | ✅ 50 epoch；5850/5850 | ✅ validation 定位：2400／3600／5850；test：5850 | ✅ validation＋test 定位 loc_metrics；生成不适用 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_loc_single/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_loc_single/experiment_plan.json) |
| G0 | `exp32_3maps_gen_single` | ✅ 50 epoch；5850/5850 | ✅ validation 离散生成：2400／3600／5850；test 离散＋连续：5850 | ✅ validation 离散、test 离散＋连续三图汇总；定位不适用 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_gen_single/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_gen_single/experiment_plan.json) |
| J0 | `exp32_3maps_joint_original` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_original/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_original/experiment_plan.json) |
| J1 | `exp32_3maps_joint_headlr` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_headlr/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_headlr/experiment_plan.json) |
| J2 | `exp32_3maps_joint_sharedlr` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_sharedlr/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_sharedlr/experiment_plan.json) |
| J3 | `exp32_3maps_joint_constloc` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_constloc/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_constloc/experiment_plan.json) |
| J4 | `exp32_3maps_joint_sharedlr_constloc` | ✅ 50 epoch；5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、3600、4800、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_sharedlr_constloc/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_sharedlr_constloc/experiment_plan.json) |
| J5 | `exp32_3maps_joint_original_constloc` | ✅ 远端完成；本地状态核实 50 epoch、5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 本地模型：1200、2400、3600、4800、5850 及根目录；已评测 2400、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_original_constloc/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_original_constloc/experiment_plan.json) |
| J6 | `exp32_3maps_joint_constloc1` | ✅ 远端完成；本地状态核实 50 epoch、5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、5850 模型文件已到；其余同步中 | [状态](outputs/csgo_1b/exp32_3maps_joint_constloc1/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_constloc1/experiment_plan.json) |
| J7 | `exp32_3maps_joint_constloc4` | ✅ 远端完成；本地状态核实 50 epoch、5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 2400、5850 模型文件已到；其余同步中 | [状态](outputs/csgo_1b/exp32_3maps_joint_constloc4/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_constloc4/experiment_plan.json) |
| J8 | `exp32_3maps_joint_constloc8` | ✅ 远端完成；本地状态核实 50 epoch、5850/5850 | ✅ validation 定位＋离散生成：2400／5850；test 定位＋离散＋连续：5850 | ✅ validation 定位＋离散生成；test 定位＋离散＋连续指标 | seen_validation；seed=42；每任务 1,500 条 | 本地模型：2400、5850 及根目录；其余同步中 | [状态](outputs/csgo_1b/exp32_3maps_joint_constloc8/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_constloc8/experiment_plan.json) |
| J9 | `exp32_3maps_joint_constloc12` | ✅ 远端完成；本地状态核实 50 epoch、5850/5850 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 本地模型：1200、2400、3600、4800、5850 及根目录；已评测 2400、5850 | [状态](outputs/csgo_1b/exp32_3maps_joint_constloc12/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_joint_constloc12/experiment_plan.json) |
| J10 | `exp32_3maps_joint_constloc16` | 远端完成（用户确认）；本地训练元数据待同步 | ✅ 定位＋离散生成：2400／5850 | ✅ 定位 loc_metrics＋离散生成三图汇总 | seen_validation；seed=42；每任务 1,500 条 | 本地模型及元数据待同步；已评测 2400、5850 | 用户远端完成回报；本地训练证据待同步 |
| A0 | `exp32_3maps_aux_control` | 未发现本地训练记录 | 未发现本地结果 | 未发现本地指标 | 待记录 | 未发现本地 checkpoint；计划首评 3600 | — |
| A1 | `exp32_3maps_perception` | 未发现本地训练记录 | 未发现本地结果 | 未发现本地指标 | 待记录 | 未发现本地 checkpoint；计划首评 3600 | — |
| A2 | `exp32_3maps_auxloc` | 未发现本地训练记录 | 未发现本地结果 | 未发现本地指标 | 待记录 | 未发现本地 checkpoint；计划首评 3600 | — |
| A3 | `exp32_3maps_both` | 仅发现计划／配置；无完成证据 | 未发现本地结果 | 未发现本地指标 | 待记录 | 未发现本地 checkpoint；计划首评 3600 | [计划](outputs/csgo_1b/exp32_3maps_both/experiment_plan.json)／[配置](outputs/csgo_1b/exp32_3maps_both/resolved_config.yaml) |
| A1_J8 | `exp32_3maps_perception_j8` | ✅ 50 epoch；5900/5900；双卡 microbatch=4、GAS=16 | ✅ 定位＋离散生成：2400／5900 | ✅ 定位宏平均＋离散生成三图汇总 | seen_validation；seed=42；每任务三图各 500 条 | 本地模型：2400、5900 及根目录；3600 未见本地评测 | [状态](outputs/csgo_1b/exp32_3maps_perception_j8/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_perception_j8/experiment_plan.json) |
| A2_J8 | `exp32_3maps_auxloc_j8` | 已配置／未启动；50 epoch horizon、首停 3600 | 待评测 | 待评测 | 计划 seen_validation；seed=42；每任务 1,500 条 | 计划首评 3600；无训练 checkpoint | [配置](csgo_configs/exp32_3maps_auxloc_j8.yaml) |
| A3_J8 | `exp32_3maps_both_j8` | 已配置／未启动；50 epoch horizon、首停 3600 | 待评测 | 待评测 | 计划 seen_validation；seed=42；每任务 1,500 条 | 计划首评 3600；无训练 checkpoint | [配置](csgo_configs/exp32_3maps_both_j8.yaml) |
| S0_J8 | `exp32_3maps_stage2_control_j8` | ✅ Stage 2 local 1800/1800；epoch=15.3846；GPU×microbatch×GAS=2×64×1 | ✅ validation 定位＋离散生成：local 1200／1800；test 定位＋离散＋连续：local 1800 | ✅ validation 定位＋离散生成；test 定位＋离散＋连续指标 | seen_validation；seed=42；每任务三图各 500 条 | 本地模型：local 1200、1800 及根目录；600 未见评测产物 | [状态](outputs/csgo_1b/exp32_3maps_stage2_control_j8/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_stage2_control_j8/experiment_plan.json) |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | ✅ Stage 2 local 1800/1800；epoch=15.2559；GPU×microbatch×GAS=2×8×8 | ✅ validation 定位＋离散生成：local 1200／1800；test 定位＋离散＋连续：local 1800 | ✅ validation 定位＋离散生成；test 定位＋离散＋连续指标 | seen_validation；seed=42；每任务三图各 500 条 | 本地模型：local 1200、1800 及根目录；600 未见评测产物 | [状态](outputs/csgo_1b/exp32_3maps_stage2_perception_j8/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_stage2_perception_j8/experiment_plan.json) |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | ✅ Stage 2 local 1800/1800；epoch=15.2559；GPU×microbatch×GAS=2×8×8 | ✅ validation 定位＋离散生成：local 1200／1800；test 定位＋离散＋连续：local 1800 | ✅ validation 定位＋离散生成；test 定位＋离散＋连续指标 | seen_validation；seed=42；每任务三图各 500 条 | 本地模型：local 1200、1800 及根目录；600 未见评测产物 | [状态](outputs/csgo_1b/exp32_3maps_stage2_auxloc_j8/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_stage2_auxloc_j8/experiment_plan.json) |
| S3_J8 | `exp32_3maps_stage2_both_j8` | ✅ Stage 2 local 1800/1800；epoch=15.2560；GPU×microbatch×GAS=2×4×16 | ✅ validation 定位＋离散生成：local 1200／1800；test 定位＋离散＋连续：local 1800 | ✅ validation 定位＋离散生成；test 定位＋离散＋连续指标 | seen_validation；seed=42；每任务三图各 500 条 | 本地模型：local 1200、1800 及根目录；600 未见评测产物 | [状态](outputs/csgo_1b/exp32_3maps_stage2_both_j8/trainer_state.json)／[计划](outputs/csgo_1b/exp32_3maps_stage2_both_j8/experiment_plan.json) |

- `checkpoint steps` 列对已同步的节点列出本地模型文件；对仍在同步的 J10 则明确标明评测节点与本地模型状态。J5～J9 所列模型仅核对 safetensors 头部声明的长度与文件大小吻合，未做校验和或 optimizer、scheduler、RNG 等完整恢复文件核查；本地模型存在也不等于每个节点都已评测。原第一轮七组均有 3600 checkpoint，但当前仅 L0、G0 有 3600 validation 结果；J4 的 4800 checkpoint 也暂无本地评测结果。未列出的保存节点不据此推断从未生成。
- 已同步的定位与离散生成均在三图各 500 条 `seen_validation` 上完成：定位逐样本文件已核对每图 500 条、共 1,500 个唯一 `(map, file_frame)`；生成汇总逐图 `Common_Count=500`、`Coverage_GT=Coverage_Pred=1`。
- 定位的 25 份 `loc_metrics.json` 已包含逐图指标及 `metrics_macro_map`，下表直接使用已有宏平均；正式定位摘要文件名为 `benchmark_csgo_v2_loc.json`。生成的 25 份 `discrete/summary.json` 已存在。J5～J10 的评测结果可核实评测节点，不代表相应模型 checkpoint 已全部同步到本地。
- J5～J10 已同步的 validation 节点为 2400 和 5850；当前未发现 3600 的本地评测结果，不据此推断该步表现。
- A3 现有计划使用 J4 配方、单 GPU、microbatch=4、GAS=32、seed=42；缺少 Trainer 状态和 checkpoint，仅登记为配置／计划已存在。A0 尚无独立记录，也未将任何第一轮结果自动复制为 A0；若后续确认复用，须注明来源及相同步数、实际配置。2026-10-14 新增的 A1_J8～A3_J8 当时仅完成配置；2026-10-22 已核对 A1_J8 完成情况并更新对应行，A2_J8／A3_J8 仍保留原状态，本次不据此推断其远端进度。
- 截至 2026-10-22，尚未发现当时已核对实验的本地 test 定位、test 离散生成或连续生成结果；本节上方旧指标均为 validation，不混入历史 exp31 三图测试结果。2026-10-26 新九组中 P1／P2／U2 的 test 结果见下方独立表；2026-10-27 又同步 L0／G0／J8／S0～S3 及 U1／P3 的 test，共 12 组，见 [test结果总表](#test-overall-20261027)。不将 test 回填为 validation。

## exp32_3maps 实验结果表

表中数值均取源 JSON 的 `metrics_macro_map`，对 `de_ancient`、`de_dust2`、`de_nuke` 做等地图宏平均，保留三位小数；全部使用 inference seed=42。不同 checkpoint 单独成行，按配置矩阵的实验顺序、各实验内步数递增排列。联合模型同一步数的两个任务对应同一 checkpoint。源文件链接保留完整精度和逐图结果；不将不同步数、不同 split 的指标合并比较。

J5～J10 已同步 2400 和 5850 的定位与离散生成 validation 指标；两任务按相同实验和步数配对。A1_J8 新增 2400／5900 的两任务结果，3600 保留缺项；最终节点按实际 Trainer 状态记为 5900，不改写为 5850。2026-10-22 新增 Stage 2 四组 local 1200／1800 结果；分别对应两阶段累计 7050／7650 optimizer steps，**1200 是预先指定的主要比较点，1800 是补充比较点**。

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
| J5 | `exp32_3maps_joint_original_constloc` | seen_validation | 2400 | 126.706 | 6.907 | 2.793 | 61.391 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_original_constloc/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J5 | `exp32_3maps_joint_original_constloc` | seen_validation | 5850 | 82.201 | 4.131 | 2.666 | 35.574 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_original_constloc/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J6 | `exp32_3maps_joint_constloc1` | seen_validation | 2400 | 134.598 | 6.843 | 2.803 | 60.586 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc1/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J6 | `exp32_3maps_joint_constloc1` | seen_validation | 5850 | 89.705 | 4.530 | 2.736 | 36.633 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc1/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J7 | `exp32_3maps_joint_constloc4` | seen_validation | 2400 | 88.933 | 4.986 | 2.710 | 39.281 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc4/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J7 | `exp32_3maps_joint_constloc4` | seen_validation | 5850 | 57.560 | 3.188 | 2.588 | 28.979 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc4/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J8 | `exp32_3maps_joint_constloc8` | seen_validation | 2400 | 67.344 | 4.415 | 2.504 | 35.516 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc8/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J8 | `exp32_3maps_joint_constloc8` | seen_validation | 5850 | 47.113 | 2.665 | 2.548 | 25.070 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc8/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J9 | `exp32_3maps_joint_constloc12` | seen_validation | 2400 | 61.070 | 3.985 | 4.047 | 33.191 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc12/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J9 | `exp32_3maps_joint_constloc12` | seen_validation | 5850 | 44.067 | 2.907 | 2.421 | 24.334 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc12/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| J10 | `exp32_3maps_joint_constloc16` | seen_validation | 2400 | 130.298 | 6.710 | 5.462 | 57.129 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc16/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| J10 | `exp32_3maps_joint_constloc16` | seen_validation | 5850 | 82.350 | 4.247 | 2.769 | 36.257 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc16/checkpoint-5850/validation/seed_42/loc_metrics.json) |
| A1_J8 | `exp32_3maps_perception_j8` | seen_validation | 2400 | 102.590 | 5.741 | 3.514 | 45.825 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_perception_j8/checkpoint-2400/validation/seed_42/loc_metrics.json) |
| A1_J8 | `exp32_3maps_perception_j8` | seen_validation | 5900 | 64.605 | 3.598 | 2.578 | 30.278 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_perception_j8/checkpoint-5900/validation/seed_42/loc_metrics.json) |
| A2_J8 | `exp32_3maps_auxloc_j8` | seen_validation（待评测） | 5900（计划） |  |  |  |  | —（已配置／未启动） |
| A3_J8 | `exp32_3maps_both_j8` | seen_validation（待评测） | 5900（计划） |  |  |  |  | —（已配置／未启动） |
| S0_J8 | `exp32_3maps_stage2_control_j8` | seen_validation | 1200（Stage 2 local） | 47.016 | 2.796 | 2.410 | 24.559 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_control_j8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| S0_J8 | `exp32_3maps_stage2_control_j8` | seen_validation | 1800（Stage 2 local） | 45.243 | 2.642 | 2.460 | 24.058 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_control_j8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | seen_validation | 1200（Stage 2 local） | 47.389 | 2.672 | 2.483 | 24.755 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_perception_j8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | seen_validation | 1800（Stage 2 local） | 45.275 | 2.598 | 2.473 | 24.254 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_perception_j8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | seen_validation | 1200（Stage 2 local） | 47.943 | 2.806 | 2.190 | 23.810 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_auxloc_j8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | seen_validation | 1800（Stage 2 local） | 44.773 | 2.673 | 2.348 | 23.847 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_auxloc_j8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| S3_J8 | `exp32_3maps_stage2_both_j8` | seen_validation | 1200（Stage 2 local） | 48.228 | 2.732 | 2.509 | 24.649 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_both_j8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| S3_J8 | `exp32_3maps_stage2_both_j8` | seen_validation | 1800（Stage 2 local） | 44.358 | 2.591 | 2.431 | 24.088 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_both_j8/checkpoint-1800/validation/seed_42/loc_metrics.json) |

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
| J5 | `exp32_3maps_joint_original_constloc` | seen_validation | 2400 | 13.361 | 0.397 | 0.657 | 0.498 | 65.284 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_original_constloc/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J5 | `exp32_3maps_joint_original_constloc` | seen_validation | 5850 | 14.518 | 0.420 | 0.607 | 0.524 | 59.700 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_original_constloc/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J6 | `exp32_3maps_joint_constloc1` | seen_validation | 2400 | 13.356 | 0.409 | 0.655 | 0.496 | 68.972 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc1/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J6 | `exp32_3maps_joint_constloc1` | seen_validation | 5850 | 14.607 | 0.421 | 0.603 | 0.528 | 61.294 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc1/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J7 | `exp32_3maps_joint_constloc4` | seen_validation | 2400 | 13.227 | 0.405 | 0.659 | 0.496 | 64.834 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc4/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J7 | `exp32_3maps_joint_constloc4` | seen_validation | 5850 | 14.198 | 0.416 | 0.621 | 0.518 | 62.457 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc4/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J8 | `exp32_3maps_joint_constloc8` | seen_validation | 2400 | 13.420 | 0.396 | 0.660 | 0.495 | 65.713 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc8/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J8 | `exp32_3maps_joint_constloc8` | seen_validation | 5850 | 14.222 | 0.416 | 0.623 | 0.517 | 61.569 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc8/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J9 | `exp32_3maps_joint_constloc12` | seen_validation | 2400 | 13.204 | 0.388 | 0.674 | 0.495 | 65.632 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc12/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J9 | `exp32_3maps_joint_constloc12` | seen_validation | 5850 | 14.087 | 0.416 | 0.638 | 0.501 | 64.164 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc12/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| J10 | `exp32_3maps_joint_constloc16` | seen_validation | 2400 | 13.346 | 0.393 | 0.666 | 0.496 | 69.046 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc16/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| J10 | `exp32_3maps_joint_constloc16` | seen_validation | 5850 | 14.498 | 0.421 | 0.612 | 0.521 | 62.497 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc16/checkpoint-5850/validation/seed_42/discrete/summary.json) |
| A1_J8 | `exp32_3maps_perception_j8` | seen_validation | 2400 | 12.885 | 0.381 | 0.664 | 0.500 | 68.275 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_perception_j8/checkpoint-2400/validation/seed_42/discrete/summary.json) |
| A1_J8 | `exp32_3maps_perception_j8` | seen_validation | 5900 | 14.197 | 0.411 | 0.625 | 0.511 | 62.819 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_perception_j8/checkpoint-5900/validation/seed_42/discrete/summary.json) |
| A2_J8 | `exp32_3maps_auxloc_j8` | seen_validation（待评测） | 5900（计划） |  |  |  |  |  | —（已配置／未启动） |
| A3_J8 | `exp32_3maps_both_j8` | seen_validation（待评测） | 5900（计划） |  |  |  |  |  | —（已配置／未启动） |
| S0_J8 | `exp32_3maps_stage2_control_j8` | seen_validation | 1200（Stage 2 local） | 14.232 | 0.412 | 0.619 | 0.520 | 63.491 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_control_j8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| S0_J8 | `exp32_3maps_stage2_control_j8` | seen_validation | 1800（Stage 2 local） | 14.337 | 0.416 | 0.617 | 0.522 | 62.816 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_control_j8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | seen_validation | 1200（Stage 2 local） | 14.348 | 0.419 | 0.619 | 0.520 | 62.466 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_perception_j8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | seen_validation | 1800（Stage 2 local） | 14.376 | 0.417 | 0.616 | 0.521 | 63.214 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_perception_j8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | seen_validation | 1200（Stage 2 local） | 14.381 | 0.421 | 0.620 | 0.518 | 64.154 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_auxloc_j8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | seen_validation | 1800（Stage 2 local） | 14.370 | 0.417 | 0.615 | 0.521 | 62.791 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_auxloc_j8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| S3_J8 | `exp32_3maps_stage2_both_j8` | seen_validation | 1200（Stage 2 local） | 14.336 | 0.418 | 0.618 | 0.518 | 64.313 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_both_j8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| S3_J8 | `exp32_3maps_stage2_both_j8` | seen_validation | 1800（Stage 2 local） | 14.353 | 0.418 | 0.617 | 0.520 | 62.596 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_both_j8/checkpoint-1800/validation/seed_42/discrete/summary.json) |

### 测试集与第二轮结果记录

| 范围 | split | 结果状态 |
|---|---|---|
| 定位测试 | seen_discrete_test | 2026-10-27 已有 L0／J8@5850、S0～S3 与 P1／U1／P2／P3／U2@local1800 的定位 test；G0 定位不适用，见[test结果总表](#test-overall-20261027) |
| 离散生成测试 | seen_discrete_test | 2026-10-27 已有 G0／J8@5850、S0～S3 与 P1／U1／P2／P3／U2@local1800 的离散生成 test；L0 生成不适用，见[test结果总表](#test-overall-20261027) |
| 连续生成测试 | seen_continuous | 2026-10-27 已有 G0／J8@5850、S0～S3 与 P1／U1／P2／P3／U2@local1800 的连续生成 test；L0 生成不适用，见[test结果总表](#test-overall-20261027) |
| A0～A3 辅助 loss 消融 | seen_validation | 尚无完成的本地评测结果；训练与配置状态见上方实验记录表 |
| A1_J8 辅助 loss 消融 | seen_validation | 2026-10-22 核对 2400／5900 定位＋离散生成指标齐全；3600 未发现本地结果 |
| A2_J8／A3_J8 辅助 loss 消融 | seen_validation | 保留原已配置／待评测记录，本次未更新；结果留空 |
| S0_J8～S3_J8 两阶段微调 | seen_validation | 2026-10-22 核对 local 1200／1800 的定位＋离散生成 validation 齐全；local600 未发现本地结果；2026-10-27 补齐 local1800 定位／离散／连续 test |

<a id="nine-ablation-results-20261026"></a>

## J8 Stage 2 九组新消融

以下矩阵以 2026-10-23 九组设计为基础；**2026-10-28 修订尚无本地训练产物的 B1／P4／U4 三行配方**，其余六行及历史结果不变，计划运行名总数仍为 35。B1 组合 P3 的 perception 0.05 ramp 与 U2 的 aux-loc 1 ramp（两项均在 local `[0,120]` 从 0 升至终值），保留 `attention=true`、`candidate_tau=1`；P4 完整沿用 P3，只改 `attention=false`；U4 完整沿用 U2，只改 `candidate_tau=0.001`。名称中的 S3／S1／S2 仅保留运行身份，不代表继续使用原高权重配方。上方 S0_J8～S3_J8 的已完成结果与历史每 300 步保存的配方不改。P3／U2 是配方参考，模型初始化始终是 J8@5850 权重，而非 P3／U2 或 S1／S2／S3 的最终权重。九组首次训练新建 optimizer、scheduler、Trainer/RNG，统一 αloc=8、原 Stage 2 分组 LR、seed=42 和有效 batch 128；`max_steps=1800`、默认完整训练、每 600 local steps 保存完整 checkpoint。只在 local 1200／1800 做 validation 的定位与离散生成；local 1800 test 运行定位、离散生成与连续生成。原计划先手动执行前七组并人工查看结果，再手动决定执行后两组；没有指标触发的自动队列。执行命令见 [record.md](record.md)，设计说明见 [Stage 2 设计文档](CSGO_BENCHMARK_V2_EXP32_3MAPS_STAGE2.md)。

| 批次 | ID | Experiment | 参考配方 | 本组变化 | 保存 | validation | test |
|---|---|---|---|---|---|---|---|
| 先七组 | P1_S1_J8 | `exp32_3maps_P1_S1_J8` | S1_J8 | perception 从 step 0 恒定 0.1；aux-loc 关 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |
| 先七组 | U1_S2_J8 | `exp32_3maps_U1_S2_J8` | S2_J8 | aux-loc 从 step 0 恒定 2；perception 关 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |
| 先七组 | B1_S3_J8 | `exp32_3maps_B1_S3_J8` | P3_S1_J8＋U2_S2_J8 | perception `[0,120] → [0,0.05]`、aux-loc `[0,120] → [0,1]`；attention=true、candidate_tau=1 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |
| 先七组 | P2_S1_J8 | `exp32_3maps_P2_S1_J8` | S1_J8 | perception `[0,120] → [0,0.2]`；aux-loc 关 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |
| 先七组 | P3_S1_J8 | `exp32_3maps_P3_S1_J8` | S1_J8 | perception `[0,120] → [0,0.05]`；aux-loc 关 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |
| 先七组 | U2_S2_J8 | `exp32_3maps_U2_S2_J8` | S2_J8 | aux-loc `[0,120] → [0,1]`；perception 关 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |
| 先七组 | U3_S2_J8 | `exp32_3maps_U3_S2_J8` | S2_J8 | aux-loc `[0,120] → [0,4]`；perception 关 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |
| 后两组 | P4_S1_J8 | `exp32_3maps_P4_S1_J8` | P3_S1_J8 | 完整沿用 P3：perception `[0,120] → [0,0.05]`、aux-loc 关；仅 attention=false | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |
| 后两组 | U4_S2_J8 | `exp32_3maps_U4_S2_J8` | U2_S2_J8 | 完整沿用 U2：aux-loc `[0,120] → [0,1]`、perception 关；仅 candidate_tau=0.001 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |

### 九组进度记录

2026-10-26 本地核对时三组完整，U1／P3 仅部分 checkpoint。2026-10-27 同步后，九组中 P1／U1／P2／P3／U2 共五组完成 local1800，local1200／1800 定位＋离散生成 validation 及 local1800 定位／离散／连续 test 均齐全；U3 仍仅计划与配置，B1／P4／U4 仍无本地训练输出。进度仅表示本地证据，不推断远端状态。已完成五组的计划均记载 seed=42、单卡 microbatch=4、GAS=32、max_steps=1800；U1 从自身 local1200、P3 从自身 local600 checkpoint 恢复，首次初始化均为 J8@5850 模型权重。

| ID | Experiment | 训练 | validation 1200 | validation 1800 | test 1800 | checkpoint／证据 |
|---|---|---|---|---|---|---|
| P1_S1_J8 | `exp32_3maps_P1_S1_J8` | ✅ local 1800/1800；单卡 microbatch=4、GAS=32 | ✅ 定位＋离散生成 | ✅ 定位＋离散生成 | ✅ 定位＋离散＋连续生成 | 本地模型 local 600／1200／1800；[状态](outputs/csgo_1b/exp32_3maps_P1_S1_J8/trainer_state.json)；[计划](outputs/csgo_1b/exp32_3maps_P1_S1_J8/experiment_plan.json) |
| U1_S2_J8 | `exp32_3maps_U1_S2_J8` | ✅ local 1800/1800；epoch=15.256；单卡 microbatch=4、GAS=32；从本组 local1200 恢复 | ✅ 定位＋离散生成 | ✅ 定位＋离散生成 | ✅ 定位＋离散＋连续生成 | 本地模型 local 600／1200／1800；[1800 状态](outputs/csgo_1b/exp32_3maps_U1_S2_J8/checkpoint-1800/trainer_state.json)；[计划](outputs/csgo_1b/exp32_3maps_U1_S2_J8/experiment_plan.json) |
| B1_S3_J8 | `exp32_3maps_B1_S3_J8` | 本地无训练输出目录；远端状态未核实 | 本地评测未同步 | 本地评测未同步 | 本地评测未同步 | 本地无 checkpoint；[静态配置](csgo_configs/exp32_3maps_B1_S3_J8.yaml) |
| P2_S1_J8 | `exp32_3maps_P2_S1_J8` | ✅ local 1800/1800；单卡 microbatch=4、GAS=32 | ✅ 定位＋离散生成 | ✅ 定位＋离散生成 | ✅ 定位＋离散＋连续生成 | 本地模型 local 600／1200／1800；[状态](outputs/csgo_1b/exp32_3maps_P2_S1_J8/trainer_state.json)；[计划](outputs/csgo_1b/exp32_3maps_P2_S1_J8/experiment_plan.json) |
| P3_S1_J8 | `exp32_3maps_P3_S1_J8` | ✅ local 1800/1800；epoch=15.256；单卡 microbatch=4、GAS=32；从本组 local600 恢复 | ✅ 定位＋离散生成 | ✅ 定位＋离散生成 | ✅ 定位＋离散＋连续生成 | 本地模型 local 600／1200／1800；[1800 状态](outputs/csgo_1b/exp32_3maps_P3_S1_J8/checkpoint-1800/trainer_state.json)；[计划](outputs/csgo_1b/exp32_3maps_P3_S1_J8/experiment_plan.json) |
| U2_S2_J8 | `exp32_3maps_U2_S2_J8` | ✅ local 1800/1800；单卡 microbatch=4、GAS=32 | ✅ 定位＋离散生成 | ✅ 定位＋离散生成 | ✅ 定位＋离散＋连续生成 | 本地模型 local 600／1200／1800；[状态](outputs/csgo_1b/exp32_3maps_U2_S2_J8/trainer_state.json)；[计划](outputs/csgo_1b/exp32_3maps_U2_S2_J8/experiment_plan.json) |
| U3_S2_J8 | `exp32_3maps_U3_S2_J8` | 仅计划／配置；无 checkpoint 或完成状态 | 本地评测未同步 | 本地评测未同步 | 本地评测未同步 | 本地无 checkpoint；[计划](outputs/csgo_1b/exp32_3maps_U3_S2_J8/experiment_plan.json) |
| P4_S1_J8 | `exp32_3maps_P4_S1_J8` | 本地无训练输出目录；远端状态未核实 | 本地评测未同步 | 本地评测未同步 | 本地评测未同步 | 本地无 checkpoint；[静态配置](csgo_configs/exp32_3maps_P4_S1_J8.yaml) |
| U4_S2_J8 | `exp32_3maps_U4_S2_J8` | 本地无训练输出目录；远端状态未核实 | 本地评测未同步 | 本地评测未同步 | 本地评测未同步 | 本地无 checkpoint；[静态配置](csgo_configs/exp32_3maps_U4_S2_J8.yaml) |

### 九组定位 validation（seen_validation；seed=42；三图各 500）

三图等权宏平均取自各源 JSON 的 `metrics_macro_map`，保留三位小数；空值表示本地结果未同步。

| ID | Experiment | Stage 2 local step | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 源文件 |
|---|---|---|---|---|---|---|---|
| P1_S1_J8 | `exp32_3maps_P1_S1_J8` | 1200 | 45.505 | 2.706 | 2.528 | 24.856 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P1_S1_J8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| P1_S1_J8 | `exp32_3maps_P1_S1_J8` | 1800 | 46.943 | 2.743 | 2.632 | 24.436 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P1_S1_J8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| U1_S2_J8 | `exp32_3maps_U1_S2_J8` | 1200 | 45.269 | 2.694 | 2.446 | 24.060 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U1_S2_J8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| U1_S2_J8 | `exp32_3maps_U1_S2_J8` | 1800 | 44.178 | 2.603 | 2.543 | 23.610 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U1_S2_J8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| B1_S3_J8 | `exp32_3maps_B1_S3_J8` | 1200 | — | — | — | — | 本地评测未同步 |
| B1_S3_J8 | `exp32_3maps_B1_S3_J8` | 1800 | — | — | — | — | 本地评测未同步 |
| P2_S1_J8 | `exp32_3maps_P2_S1_J8` | 1200 | 45.135 | 2.708 | 2.635 | 24.198 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P2_S1_J8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| P2_S1_J8 | `exp32_3maps_P2_S1_J8` | 1800 | 45.049 | 2.653 | 2.431 | 24.061 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P2_S1_J8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| P3_S1_J8 | `exp32_3maps_P3_S1_J8` | 1200 | 47.813 | 2.744 | 2.651 | 24.216 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P3_S1_J8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| P3_S1_J8 | `exp32_3maps_P3_S1_J8` | 1800 | 45.031 | 2.645 | 2.406 | 23.977 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P3_S1_J8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| U2_S2_J8 | `exp32_3maps_U2_S2_J8` | 1200 | 45.013 | 2.697 | 2.710 | 24.745 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U2_S2_J8/checkpoint-1200/validation/seed_42/loc_metrics.json) |
| U2_S2_J8 | `exp32_3maps_U2_S2_J8` | 1800 | 43.858 | 2.611 | 2.533 | 24.263 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U2_S2_J8/checkpoint-1800/validation/seed_42/loc_metrics.json) |
| U3_S2_J8 | `exp32_3maps_U3_S2_J8` | 1200 | — | — | — | — | 本地评测未同步 |
| U3_S2_J8 | `exp32_3maps_U3_S2_J8` | 1800 | — | — | — | — | 本地评测未同步 |
| P4_S1_J8 | `exp32_3maps_P4_S1_J8` | 1200 | — | — | — | — | 本地评测未同步 |
| P4_S1_J8 | `exp32_3maps_P4_S1_J8` | 1800 | — | — | — | — | 本地评测未同步 |
| U4_S2_J8 | `exp32_3maps_U4_S2_J8` | 1200 | — | — | — | — | 本地评测未同步 |
| U4_S2_J8 | `exp32_3maps_U4_S2_J8` | 1800 | — | — | — | — | 本地评测未同步 |

### 九组离散生成 validation（seen_validation；seed=42；三图各 500）

三图等权宏平均取自各源 JSON 的 `metrics_macro_map`，保留三位小数；空值表示本地结果未同步。

| ID | Experiment | Stage 2 local step | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 源文件 |
|---|---|---|---|---|---|---|---|---|
| P1_S1_J8 | `exp32_3maps_P1_S1_J8` | 1200 | 14.376 | 0.418 | 0.615 | 0.521 | 62.817 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P1_S1_J8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| P1_S1_J8 | `exp32_3maps_P1_S1_J8` | 1800 | 14.370 | 0.419 | 0.614 | 0.522 | 61.776 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P1_S1_J8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| U1_S2_J8 | `exp32_3maps_U1_S2_J8` | 1200 | 14.316 | 0.420 | 0.617 | 0.518 | 63.155 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U1_S2_J8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| U1_S2_J8 | `exp32_3maps_U1_S2_J8` | 1800 | 14.376 | 0.418 | 0.615 | 0.521 | 63.152 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U1_S2_J8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| B1_S3_J8 | `exp32_3maps_B1_S3_J8` | 1200 | — | — | — | — | — | 本地评测未同步 |
| B1_S3_J8 | `exp32_3maps_B1_S3_J8` | 1800 | — | — | — | — | — | 本地评测未同步 |
| P2_S1_J8 | `exp32_3maps_P2_S1_J8` | 1200 | 14.272 | 0.415 | 0.618 | 0.519 | 63.766 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P2_S1_J8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| P2_S1_J8 | `exp32_3maps_P2_S1_J8` | 1800 | 14.361 | 0.414 | 0.615 | 0.521 | 62.558 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P2_S1_J8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| P3_S1_J8 | `exp32_3maps_P3_S1_J8` | 1200 | 14.368 | 0.420 | 0.617 | 0.518 | 63.110 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P3_S1_J8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| P3_S1_J8 | `exp32_3maps_P3_S1_J8` | 1800 | 14.455 | 0.421 | 0.615 | 0.521 | 62.625 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P3_S1_J8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| U2_S2_J8 | `exp32_3maps_U2_S2_J8` | 1200 | 14.345 | 0.419 | 0.617 | 0.519 | 63.259 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U2_S2_J8/checkpoint-1200/validation/seed_42/discrete/summary.json) |
| U2_S2_J8 | `exp32_3maps_U2_S2_J8` | 1800 | 14.364 | 0.418 | 0.615 | 0.521 | 62.206 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U2_S2_J8/checkpoint-1800/validation/seed_42/discrete/summary.json) |
| U3_S2_J8 | `exp32_3maps_U3_S2_J8` | 1200 | — | — | — | — | — | 本地评测未同步 |
| U3_S2_J8 | `exp32_3maps_U3_S2_J8` | 1800 | — | — | — | — | — | 本地评测未同步 |
| P4_S1_J8 | `exp32_3maps_P4_S1_J8` | 1200 | — | — | — | — | — | 本地评测未同步 |
| P4_S1_J8 | `exp32_3maps_P4_S1_J8` | 1800 | — | — | — | — | — | 本地评测未同步 |
| U4_S2_J8 | `exp32_3maps_U4_S2_J8` | 1200 | — | — | — | — | — | 本地评测未同步 |
| U4_S2_J8 | `exp32_3maps_U4_S2_J8` | 1800 | — | — | — | — | — | 本地评测未同步 |


<a id="u5-aux05-20261027"></a>

## U5_S2_J8 aux-loc 终值 0.5 补充（未启动）

U5 是九组之后的独立补充，不改写 2026-10-23 九组矩阵或 2026-10-26 的历史结果。它完整继承 U2_S2_J8 配方，唯一训练差异是 aux-loc loss 权重的 local `[0,120] → [0,1]` 改为 `[0,120] → [0,0.5]`，之后恒定 0.5；perception 关闭。U2 仅作为配置参考，U5 首次训练只加载 J8@5850 的 `model.safetensors`，从 local step 0 新建 optimizer、scheduler、Trainer/RNG，绝非从 U2@1800 续训。[静态配置](csgo_configs/exp32_3maps_U5_S2_J8.yaml)、[Stage 2 设计说明](CSGO_BENCHMARK_V2_EXP32_3MAPS_STAGE2.md#u5-aux05-20261027)与 [record.md 手动命令](record.md#exp32_3maps-u5_s2_j8-2026-10-27-手动命令)记录了来源和执行方式。

| ID | Experiment | 参考配方 | 唯一变化 | 固定训练设置 | 保存 | validation | test |
|---|---|---|---|---|---|---|---|
| U5_S2_J8 | `exp32_3maps_U5_S2_J8` | U2_S2_J8 | aux-loc `[0,120] → [0,0.5]`，之后恒定；perception 关 | paired joint；αloc=8；普通组峰值 LR 2e-5，Action connector/norm/projector 1e-4；warmup 60；max_steps=1800；seed=42；单卡 microbatch=4、有效 batch=128 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |

| ID | 训练 | validation 1200 | validation 1800 | test 1800 | 证据 |
|---|---|---|---|---|---|
| U5_S2_J8 | 未启动；无 checkpoint | 待评 | 待评 | 待评 | 仅[静态配置](csgo_configs/exp32_3maps_U5_S2_J8.yaml)；不得将 U2 结果记为 U5 |

以下均为待填占位，`—` 表示没有结果，不代表 0。validation 使用 seen_validation、seed=42、三图各 500 条和同一 checkpoint 的定位／离散生成结果；所有步数均为 Stage 2 local step。

### U5 定位 validation

| ID | local step | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 源文件 |
|---|---:|---:|---:|---:|---:|---|
| U5_S2_J8 | 1200 | — | — | — | — | 待评 |
| U5_S2_J8 | 1800 | — | — | — | — | 待评 |

### U5 离散生成 validation

| ID | local step | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 源文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| U5_S2_J8 | 1200 | — | — | — | — | — | 待评 |
| U5_S2_J8 | 1800 | — | — | — | — | — | 待评 |

### U5 test local 1800

test 仅在 local1800 执行：seen_discrete_test 定位／离散生成各图 2,000 条，seen_continuous 连续生成各图 20×64 帧；seed=42，结果用于报告，不用于回选配置。

| ID | 定位 test | 离散生成 test | 连续生成 test | 源文件 |
|---|---|---|---|---|
| U5_S2_J8 | 待评 | 待评 | 待评 | 待同步 |

<a id="p5-perception025-20261028"></a>

## P5_S1_J8 perception 终值 0.025 补充（未启动）

P5 是原九组与 U5 之外的独立补充，不改写原九组矩阵或历史结果。它完整继承 P3_S1_J8 配方，唯一训练差异是 perception loss 权重的 local `[0,120] → [0,0.05]` 改为 `[0,120] → [0,0.025]`，之后恒定 0.025；保留 ramp，aux-loc 关闭。P3 仅作为配方参考；P5 首次训练只加载 J8@5850 的 `model.safetensors`，从 local step 0 新建 optimizer、scheduler、Trainer/RNG，不从 P3@1800 续训。[静态配置](csgo_configs/exp32_3maps_P5_S1_J8.yaml)、[Stage 2 设计说明](CSGO_BENCHMARK_V2_EXP32_3MAPS_STAGE2.md#p5-perception025-20261028)与 [record.md 手动命令](record.md#exp32_3maps-p5_s1_j8-2026-10-28-手动命令)记录了来源和执行方式。

| ID | Experiment | 参考配方 | 唯一变化 | 固定训练设置 | 保存 | validation | test |
|---|---|---|---|---|---|---|---|
| P5_S1_J8 | `exp32_3maps_P5_S1_J8` | P3_S1_J8 | perception `[0,120] → [0,0.025]`，之后恒定；aux-loc 关 | paired joint；αloc=8；普通组峰值 LR 2e-5，Action connector/norm/projector 1e-4；AdamW；cosine min LR `min_lr_rate=0.1`；warmup 60；max_steps=1800；seed=42；单卡 microbatch=4、目标有效 batch=128 | local 600／1200／1800 | local 1200／1800：定位＋离散生成 | local 1800：定位＋离散＋连续生成 |

| ID | 训练 | validation 1200 | validation 1800 | test 1800 | 证据 |
|---|---|---|---|---|---|
| P5_S1_J8 | 未启动；无 checkpoint | 待评 | 待评 | 待评 | 仅[静态配置](csgo_configs/exp32_3maps_P5_S1_J8.yaml)；不得将 P3 结果记为 P5 |

以下均为待填占位，`—` 表示没有结果，不代表 0；源文件待生成。validation 使用 seen_validation、seed=42、三图各 500 条和同一 checkpoint 的定位／离散生成结果；所有步数均为 Stage 2 local step。

### P5 定位 validation

| ID | local step | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 源文件 |
|---|---:|---:|---:|---:|---:|---|
| P5_S1_J8 | 1200 | — | — | — | — | 待生成 |
| P5_S1_J8 | 1800 | — | — | — | — | 待生成 |

### P5 离散生成 validation

| ID | local step | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 源文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| P5_S1_J8 | 1200 | — | — | — | — | — | 待生成 |
| P5_S1_J8 | 1800 | — | — | — | — | — | 待生成 |

### P5 test local 1800

test 仅在 local1800 执行：seen_discrete_test 定位／离散生成各图 2,000 条，seen_continuous 连续生成各图 20×64 帧；seed=42，结果用于报告，不用于回选配置。

| ID | 定位 test | 离散生成 test | 连续生成 test | 源文件 |
|---|---|---|---|---|
| P5_S1_J8 | 待评 | 待评 | 待评 | 待生成 |


## test local 1800

以下报告已完成的 12 组 test：L0／G0／J8 为单阶段 checkpoint-5850，S0_J8～S3_J8 与 P1／U1／P2／P3／U2 为 Stage 2 local checkpoint-1800。checkpoint 列明确训练节点；不同训练预算仅作同 split 描述，不视为等预算因果比较，也不与上方 validation 混比。定位与离散生成使用 `seen_discrete_test`，三图各 2,000 条；连续生成使用 `seen_continuous`，三图各 1,280 帧、每图 20 条轨迹。推理 seed=42；已填十二组的源 JSON 均核实地图、split、checkpoint、seed 与样本数；离散和连续生成还核实逐图 `Coverage_GT=Coverage_Pred=1`。数字为 `metrics_macro_map` 三图宏平均。`—` 在 L0 的生成或 G0 的定位表示该单任务模型不适用；在尚无本地结果的消融组中表示指标缺失，绝不按 0 处理。

<a id="test-overall-20261027"></a>

### test结果总表（2026-10-27）

范围为下列 12 个已完成模型；各列加粗表示在适用模型中按 JSON 完整精度确定的最优宏平均，显示值统一保留三位小数。L0／G0／J8 是单阶段 5850 步，S0～S3 与 P1／U1／P2／P3／U2 是 J8 初始化后另训 local 1800 步，预算不同。TWE = Temporal_Warping_Error，TDE = Temporal_Difference_Error，OFE = Optical_Flow_EPE。`—` 为单任务模型不适用；完整路径见下方三张明细表。

| ID | checkpoint | 定位 XY↓ | 定位 Z↓ | 定位 Pitch↓ | 定位 Yaw↓ | 离散 PSNR↑ | 离散 SSIM↑ | 离散 LPIPS↓ | 离散 Boundary_F1↑ | 离散 FID↓ | 连续 PSNR↑ | 连续 SSIM↑ | 连续 LPIPS↓ | 连续 Boundary_F1↑ | 连续 TWE↓ | 连续 TDE↓ | 连续 OFE↓ | 连续 FVD↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 5850（单阶段） | 58.757 | 3.131 | 3.048 | 31.862 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| G0 | 5850（单阶段） | — | — | — | — | 14.078 | 0.400 | **0.630** | 0.515 | 37.025 | 14.834 | 0.428 | 0.600 | 0.531 | 34.884 | 39.939 | 16.210 | 894.198 |
| J8 | 5850（单阶段） | 55.512 | 2.972 | 2.800 | 30.797 | 14.018 | 0.400 | 0.636 | 0.514 | **35.557** | 14.675 | 0.428 | 0.608 | 0.530 | 34.121 | 39.406 | 16.187 | 932.428 |
| S0_J8 | 1800（Stage 2 local） | 54.078 | 2.940 | 2.687 | 30.168 | 14.062 | 0.399 | 0.632 | 0.518 | 36.732 | 14.860 | 0.429 | 0.598 | 0.536 | 33.450 | 38.706 | 16.284 | 907.361 |
| S1_J8 | 1800（Stage 2 local） | 54.109 | 2.881 | 2.716 | 30.102 | 14.101 | 0.399 | 0.632 | 0.517 | 37.025 | 14.944 | 0.431 | 0.594 | 0.536 | 33.586 | 38.782 | 16.254 | 931.717 |
| S2_J8 | 1800（Stage 2 local） | 54.154 | 2.963 | **2.592** | **29.930** | 14.078 | 0.399 | 0.632 | 0.517 | 37.629 | 15.040 | 0.434 | 0.588 | 0.538 | 32.961 | 38.232 | 16.140 | **874.741** |
| S3_J8 | 1800（Stage 2 local） | 52.968 | **2.849** | 2.637 | 30.107 | 14.056 | 0.399 | 0.632 | 0.516 | 36.928 | 14.918 | 0.432 | 0.598 | 0.531 | 34.146 | 39.339 | 16.170 | 932.101 |
| P1_S1_J8 | 1800（local） | 55.536 | 3.064 | 2.849 | 30.190 | 14.063 | 0.400 | 0.632 | 0.516 | 36.002 | 15.015 | 0.435 | 0.589 | 0.537 | 33.978 | 39.136 | 16.168 | 902.529 |
| U1_S2_J8 | 1800（local） | 53.439 | 2.917 | 2.774 | 30.006 | 14.059 | 0.400 | 0.631 | 0.516 | 36.312 | 15.084 | 0.436 | 0.584 | 0.540 | 33.369 | 38.544 | 16.037 | 894.689 |
| P2_S1_J8 | 1800（local） | 53.527 | 2.924 | 2.690 | 30.532 | 14.028 | 0.394 | 0.632 | 0.517 | 36.117 | 15.024 | 0.432 | 0.586 | 0.538 | 33.349 | 38.536 | 16.136 | 895.861 |
| P3_S1_J8 | 1800（local） | 54.174 | 2.930 | 2.637 | 30.087 | **14.132** | **0.402** | 0.632 | 0.518 | 36.537 | **15.170** | **0.440** | **0.582** | **0.542** | **32.272** | **37.577** | **15.884** | 916.047 |
| U2_S2_J8 | 1800（local） | **52.848** | 2.890 | 2.759 | 29.938 | 14.093 | 0.399 | 0.631 | **0.518** | 37.156 | 14.947 | 0.433 | 0.594 | 0.534 | 33.707 | 38.960 | 16.158 | 896.949 |


### 定位 test（seen_discrete_test）

| ID | Experiment | checkpoint | XY_Dist↓ | Z_Dist↓ | Pitch_Dist↓ | Yaw_Dist↓ | 源文件 |
|---|---|---|---|---|---|---|---|
| L0 | `exp32_3maps_loc_single` | 5850（单阶段） | 58.757 | 3.131 | 3.048 | 31.862 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_loc_single/checkpoint-5850/test/seed_42/loc_metrics.json) |
| G0 | `exp32_3maps_gen_single` | 5850（单阶段） | — | — | — | — | 不适用（单任务模型） |
| J8 | `exp32_3maps_joint_constloc8` | 5850（单阶段） | 55.512 | 2.972 | 2.800 | 30.797 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_joint_constloc8/checkpoint-5850/test/seed_42/loc_metrics.json) |
| S0_J8 | `exp32_3maps_stage2_control_j8` | 1800（Stage 2 local） | 54.078 | 2.940 | 2.687 | 30.168 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_control_j8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | 1800（Stage 2 local） | 54.109 | 2.881 | 2.716 | 30.102 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_perception_j8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | 1800（Stage 2 local） | 54.154 | 2.963 | 2.592 | 29.930 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_auxloc_j8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| S3_J8 | `exp32_3maps_stage2_both_j8` | 1800（Stage 2 local） | 52.968 | 2.849 | 2.637 | 30.107 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_stage2_both_j8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| P1_S1_J8 | `exp32_3maps_P1_S1_J8` | 1800（local） | 55.536 | 3.064 | 2.849 | 30.190 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P1_S1_J8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| U1_S2_J8 | `exp32_3maps_U1_S2_J8` | 1800（local） | 53.439 | 2.917 | 2.774 | 30.006 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U1_S2_J8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| B1_S3_J8 | `exp32_3maps_B1_S3_J8` | 1800（计划 local） | — | — | — | — | 本地评测未同步 |
| P2_S1_J8 | `exp32_3maps_P2_S1_J8` | 1800（local） | 53.527 | 2.924 | 2.690 | 30.532 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P2_S1_J8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| P3_S1_J8 | `exp32_3maps_P3_S1_J8` | 1800（local） | 54.174 | 2.930 | 2.637 | 30.087 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_P3_S1_J8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| U2_S2_J8 | `exp32_3maps_U2_S2_J8` | 1800（local） | 52.848 | 2.890 | 2.759 | 29.938 | [loc_metrics.json](outputs_loc/benchmark_v2/exp32_3maps_U2_S2_J8/checkpoint-1800/test/seed_42/loc_metrics.json) |
| U3_S2_J8 | `exp32_3maps_U3_S2_J8` | 1800（计划 local） | — | — | — | — | 本地评测未同步 |
| P4_S1_J8 | `exp32_3maps_P4_S1_J8` | 1800（计划 local） | — | — | — | — | 本地评测未同步 |
| U4_S2_J8 | `exp32_3maps_U4_S2_J8` | 1800（计划 local） | — | — | — | — | 本地评测未同步 |

### 离散生成 test（seen_discrete_test）

| ID | Experiment | checkpoint | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | FID↓ | 源文件 |
|---|---|---|---|---|---|---|---|---|
| L0 | `exp32_3maps_loc_single` | 5850（单阶段） | — | — | — | — | — | 不适用（单任务模型） |
| G0 | `exp32_3maps_gen_single` | 5850（单阶段） | 14.078 | 0.400 | 0.630 | 0.515 | 37.025 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_gen_single/checkpoint-5850/test/seed_42/discrete/summary.json) |
| J8 | `exp32_3maps_joint_constloc8` | 5850（单阶段） | 14.018 | 0.400 | 0.636 | 0.514 | 35.557 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc8/checkpoint-5850/test/seed_42/discrete/summary.json) |
| S0_J8 | `exp32_3maps_stage2_control_j8` | 1800（Stage 2 local） | 14.062 | 0.399 | 0.632 | 0.518 | 36.732 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_control_j8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | 1800（Stage 2 local） | 14.101 | 0.399 | 0.632 | 0.517 | 37.025 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_perception_j8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | 1800（Stage 2 local） | 14.078 | 0.399 | 0.632 | 0.517 | 37.629 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_auxloc_j8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| S3_J8 | `exp32_3maps_stage2_both_j8` | 1800（Stage 2 local） | 14.056 | 0.399 | 0.632 | 0.516 | 36.928 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_both_j8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| P1_S1_J8 | `exp32_3maps_P1_S1_J8` | 1800（local） | 14.063 | 0.400 | 0.632 | 0.516 | 36.002 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P1_S1_J8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| U1_S2_J8 | `exp32_3maps_U1_S2_J8` | 1800（local） | 14.059 | 0.400 | 0.631 | 0.516 | 36.312 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U1_S2_J8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| B1_S3_J8 | `exp32_3maps_B1_S3_J8` | 1800（计划 local） | — | — | — | — | — | 本地评测未同步 |
| P2_S1_J8 | `exp32_3maps_P2_S1_J8` | 1800（local） | 14.028 | 0.394 | 0.632 | 0.517 | 36.117 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P2_S1_J8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| P3_S1_J8 | `exp32_3maps_P3_S1_J8` | 1800（local） | 14.132 | 0.402 | 0.632 | 0.518 | 36.537 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P3_S1_J8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| U2_S2_J8 | `exp32_3maps_U2_S2_J8` | 1800（local） | 14.093 | 0.399 | 0.631 | 0.518 | 37.156 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U2_S2_J8/checkpoint-1800/test/seed_42/discrete/summary.json) |
| U3_S2_J8 | `exp32_3maps_U3_S2_J8` | 1800（计划 local） | — | — | — | — | — | 本地评测未同步 |
| P4_S1_J8 | `exp32_3maps_P4_S1_J8` | 1800（计划 local） | — | — | — | — | — | 本地评测未同步 |
| U4_S2_J8 | `exp32_3maps_U4_S2_J8` | 1800（计划 local） | — | — | — | — | — | 本地评测未同步 |

### 连续生成 test（seen_continuous）

| ID | Experiment | checkpoint | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary_F1↑ | Temporal_Warping_Error↓ | Temporal_Difference_Error↓ | Optical_Flow_EPE↓ | FVD↓ | 源文件 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | `exp32_3maps_loc_single` | 5850（单阶段） | — | — | — | — | — | — | — | — | 不适用（单任务模型） |
| G0 | `exp32_3maps_gen_single` | 5850（单阶段） | 14.834 | 0.428 | 0.600 | 0.531 | 34.884 | 39.939 | 16.210 | 894.198 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_gen_single/checkpoint-5850/test/seed_42/continuous/summary.json) |
| J8 | `exp32_3maps_joint_constloc8` | 5850（单阶段） | 14.675 | 0.428 | 0.608 | 0.530 | 34.121 | 39.406 | 16.187 | 932.428 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_joint_constloc8/checkpoint-5850/test/seed_42/continuous/summary.json) |
| S0_J8 | `exp32_3maps_stage2_control_j8` | 1800（Stage 2 local） | 14.860 | 0.429 | 0.598 | 0.536 | 33.450 | 38.706 | 16.284 | 907.361 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_control_j8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| S1_J8 | `exp32_3maps_stage2_perception_j8` | 1800（Stage 2 local） | 14.944 | 0.431 | 0.594 | 0.536 | 33.586 | 38.782 | 16.254 | 931.717 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_perception_j8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| S2_J8 | `exp32_3maps_stage2_auxloc_j8` | 1800（Stage 2 local） | 15.040 | 0.434 | 0.588 | 0.538 | 32.961 | 38.232 | 16.140 | 874.741 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_auxloc_j8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| S3_J8 | `exp32_3maps_stage2_both_j8` | 1800（Stage 2 local） | 14.918 | 0.432 | 0.598 | 0.531 | 34.146 | 39.339 | 16.170 | 932.101 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_stage2_both_j8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| P1_S1_J8 | `exp32_3maps_P1_S1_J8` | 1800（local） | 15.015 | 0.435 | 0.589 | 0.537 | 33.978 | 39.136 | 16.168 | 902.529 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P1_S1_J8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| U1_S2_J8 | `exp32_3maps_U1_S2_J8` | 1800（local） | 15.084 | 0.436 | 0.584 | 0.540 | 33.369 | 38.544 | 16.037 | 894.689 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U1_S2_J8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| B1_S3_J8 | `exp32_3maps_B1_S3_J8` | 1800（计划 local） | — | — | — | — | — | — | — | — | 本地评测未同步 |
| P2_S1_J8 | `exp32_3maps_P2_S1_J8` | 1800（local） | 15.024 | 0.432 | 0.586 | 0.538 | 33.349 | 38.536 | 16.136 | 895.861 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P2_S1_J8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| P3_S1_J8 | `exp32_3maps_P3_S1_J8` | 1800（local） | 15.170 | 0.440 | 0.582 | 0.542 | 32.272 | 37.577 | 15.884 | 916.047 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_P3_S1_J8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| U2_S2_J8 | `exp32_3maps_U2_S2_J8` | 1800（local） | 14.947 | 0.433 | 0.594 | 0.534 | 33.707 | 38.960 | 16.158 | 896.949 | [summary.json](outputs_eval/benchmark_v2/exp32_3maps_U2_S2_J8/checkpoint-1800/test/seed_42/continuous/summary.json) |
| U3_S2_J8 | `exp32_3maps_U3_S2_J8` | 1800（计划 local） | — | — | — | — | — | — | — | — | 本地评测未同步 |
| P4_S1_J8 | `exp32_3maps_P4_S1_J8` | 1800（计划 local） | — | — | — | — | — | — | — | — | 本地评测未同步 |
| U4_S2_J8 | `exp32_3maps_U4_S2_J8` | 1800（计划 local） | — | — | — | — | — | — | — | — | 本地评测未同步 |

<a id="stage2-new-results-20261026"></a>

## 2026-10-26：九组新消融的阶段性结果分析

更新于 **2026-10-27**：本节纳入新同步的 U1、P3，替换原先仅含 P1／P2／U2 的阶段性判断。九组中 **P1／U1／P2／P3／U2 五组**的 local1200／1800 validation 与 local1800 test 已齐全；B1／U3／P4／U4 尚无本地完整结果，不据此推断远端进度。L0／G0／J8 与 S0～S3 的最终 test 也已同步，完整指标见 [test结果总表](#test-overall-20261027)。

**辅助目标已经带来可辨认的增量收益，但收益分布在不同任务和指标上。** P3 是当前 perception 路线较均衡的候选；U2 的 XY 收益最清楚，U1 改善部分定位分量并在连续生成上优于无辅助续训 S0。与此同时，P3 仍有 Z／边界等退化项，U1／U2 都损失 Pitch 精度。当前结果支持有条件的辅助收益，尚不支持“辅助＋联合 > 联合 > 单任务”的全面排序；是否有贡献也不应只以一个模型是否包揽全部指标最优来判断。

**比较口径。** 以下主表使用相同的 local1800 validation，local1200 仅用于观察训练走势，不拼接两个 checkpoint 的最佳指标。五组首次初始化均为 J8@5850；P1／U1 取消的是各自辅助 **loss 权重 ramp**，仍保留学习率的 60 步 warmup。L0／G0／J8 的 5850 步结果作为参照，辅助目标的增量主要对照同为 Stage 2 local1800 的 S0，单因素改变则分别对照 S1 或 S2。本节不以显卡数量、依赖版本的小幅差异或单 seed 随机性作为主要解释，也不据此安排重跑。

表内数字为源 JSON 的 `metrics_macro_map` 三图等权宏平均，保留三位小数；**加粗仅比较九个 Stage 2 模型**（S0～S3＋新五组），按完整精度确定，不包含预算不同的前三个参照。`—` 表示单任务模型不适用，原始数值与路径见上方 validation 明细表。

| 实验 | checkpoint | XY↓ | Z↓ | Pitch↓ | Yaw↓ | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 5850（单阶段） | 48.459 | 2.878 | 2.818 | 26.235 | — | — | — | — | — |
| G0 | 5850（单阶段） | — | — | — | — | 14.402 | 0.418 | 0.613 | 0.520 | 62.252 |
| J8 | 5850（单阶段） | 47.113 | 2.665 | 2.548 | 25.070 | 14.222 | 0.416 | 0.623 | 0.517 | 61.569 |
| S0_J8 | 1800（Stage 2 local） | 45.243 | 2.642 | 2.460 | 24.058 | 14.337 | 0.416 | 0.617 | **0.522** | 62.816 |
| S1_J8 | 1800（Stage 2 local） | 45.275 | 2.598 | 2.473 | 24.254 | 14.376 | 0.417 | 0.616 | 0.521 | 63.214 |
| S2_J8 | 1800（Stage 2 local） | 44.773 | 2.673 | **2.348** | 23.847 | 14.370 | 0.417 | 0.615 | 0.521 | 62.791 |
| S3_J8 | 1800（Stage 2 local） | 44.358 | **2.591** | 2.431 | 24.088 | 14.353 | 0.418 | 0.617 | 0.520 | 62.596 |
| P1_S1_J8 | 1800（Stage 2 local） | 46.943 | 2.743 | 2.632 | 24.436 | 14.370 | 0.419 | **0.614** | 0.522 | **61.776** |
| U1_S2_J8 | 1800（Stage 2 local） | 44.178 | 2.603 | 2.543 | **23.610** | 14.376 | 0.418 | 0.615 | 0.521 | 63.152 |
| P2_S1_J8 | 1800（Stage 2 local） | 45.049 | 2.653 | 2.431 | 24.061 | 14.361 | 0.414 | 0.615 | 0.521 | 62.558 |
| P3_S1_J8 | 1800（Stage 2 local） | 45.031 | 2.645 | 2.406 | 23.977 | **14.455** | **0.421** | 0.615 | 0.521 | 62.625 |
| U2_S2_J8 | 1800（Stage 2 local） | **43.858** | 2.611 | 2.533 | 24.263 | 14.364 | 0.418 | 0.615 | 0.521 | 62.206 |

**P3：perception 权重 0.1→0.05、保留 ramp，是当前更值得保留的 perception 配方。** 相比 S1@1800，validation 的 XY／Pitch／Yaw 分别下降 0.54%／2.68%／1.15%，但 Z 增加 1.81%；PSNR 从 14.376 升至 14.455，SSIM 从 0.417079 升至 0.420720，LPIPS 降低 0.001572、FID 降低 0.588，Boundary F1 降低 0.000494。相对 S0，P3 的 XY／Pitch／Yaw 及 PSNR／SSIM／LPIPS／FID 也更好，代价是 Z 增加 0.10%、Boundary F1 降低 0.001578。其 local1200→1800 的九项 validation 宏平均全部改善；最终 PSNR／SSIM 为本表 Stage 2 最优。P3 相对高权重 P2 的四项定位、PSNR／SSIM 更好，LPIPS 几乎相同，FID／Boundary F1 则略差。因此优先保留 0.05＋ramp 这一候选，无需立即扩大 perception 权重网格，也不能把它称为所有指标都最优的默认配方。

**U1：去掉 aux-loc ramp 改善空间定位与 validation Yaw，但没有消除角度分量的取舍。** 相比 S2@1800，validation 的 XY／Z／Yaw 分别下降 1.33%／2.62%／0.99%，Pitch 却增加 8.30%，且三图 Pitch 均变差。PSNR／SSIM／LPIPS 改善，FID 从 62.791 增至 63.152，Boundary F1 略降。相对 S0，U1 的 XY／Z／Yaw 下降 2.35%／1.49%／1.86%，但 Pitch 增加 3.40%；它取得当前 Stage 2 最低的 validation Yaw=23.610。local1200→1800 时 XY／Z／Yaw 继续改善，Pitch 由 2.446 增至 2.543、SSIM 也下降，不能概括为后半程所有指标都改善。U1 与 U2 同时存在 ramp 和终值两项差异，二者可用于比较最终表现，不能把它们的差值归因于单独一个超参数。

**U2：aux-loc 权重 2→1 的 XY 收益仍成立，但不意味着权重越小越好。** 相比 S2@1800，validation 的 XY 从 44.773 降至 43.858（−2.04%）、Z 下降 2.31%，Pitch／Yaw 却增加 7.87%／1.75%；XY 三图均改善，Pitch 三图均变差。相对 S0，XY 下降 3.06%，也是当前表内最低值，但 Pitch／Yaw 更差。生成相对 S0 的 PSNR／SSIM／LPIPS／FID 改善，Boundary F1 降低 0.00133。local1200→1800 的三图四项定位指标全部改善，支持保留完整 1800 步作为既定终点。U2 适合作为 XY 优先候选，S2 则继续作为 Pitch 与连续生成的重要参照；既有 U5（终值 0.5）应检验这种取舍是否继续扩大，而不是预设进一步降权一定改善整体表现。

**P1／P2：作为启用时机和较高权重的对照保留，不支持更早启用或更大权重普遍更好。** P1 去掉 perception ramp 后，相对 S1 的 validation 四项定位均变差，XY 增加 3.68%；local1200→1800 的 XY／Z／Pitch 也恶化。它仍取得当前 Stage 2 最低的 validation LPIPS／FID，但 SSIM 最优已由 P3 取得，不能再沿用原先三项生成指标都最好的说法。P2 将 perception 权重增至 0.2 后，相对 S1 的 validation XY／Pitch／Yaw 略好、Z 更差；LPIPS／FID 改善，PSNR／SSIM／Boundary F1 下降，其中 SSIM 三图均下降。P3 的结果使较低权重路线更值得保留，P1／P2 仍用于展示定位与生成之间的取舍。

**test 现在可以与旧参照在同一 split 上比较，但不用于反选超参数或 checkpoint。** 下表均比较预先指定的 local1800 test 宏平均；定位与离散生成各为三图 6,000 条，连续生成为 3,840 帧。改善或退化按 JSON 完整精度判断，逐项方向不构成跨指标总分。连续生成没有本节对应的 validation 结果，不能称为连续指标的“验证趋势复现”。

| 新组／对照 | 定位 test | 离散生成 test | 连续生成 test |
|---|---|---|---|
| P1／S1 | 四项误差均更高 | SSIM／LPIPS／FID 改善；PSNR／Boundary F1 下降 | PSNR／SSIM／LPIPS／Boundary F1／OFE／FVD 改善；TWE／TDE 更差 |
| U1／S2 | XY／Z 改善；Pitch／Yaw 更差 | SSIM／LPIPS／FID 改善；PSNR／Boundary F1 下降 | PSNR／SSIM／LPIPS／Boundary F1／OFE 改善；TWE／TDE／FVD 更差 |
| P2／S1 | XY／Pitch 改善；Z／Yaw 更差 | 仅 FID 改善，其余四项更差 | 八项宏平均均改善 |
| P3／S1 | Pitch／Yaw 改善；XY／Z 更差 | PSNR／SSIM／Boundary F1／FID 改善；LPIPS 略差 | 八项宏平均均改善 |
| U2／S2 | XY／Z 改善；Pitch／Yaw 更差 | 五项宏平均均改善 | 八项宏平均均更差 |

P3 的 test 离散 PSNR=14.132、SSIM=0.401906，以及连续 PSNR／SSIM／LPIPS／Boundary F1／TWE／TDE／OFE，为当前 **12 组总表**最优；但连续 FVD=916.047，仍差于 S0 的 907.361 和 S2 的 874.741。U1 的连续八项宏平均则全部优于 S0，包括 FVD 从 907.361 降至 894.689；同时其 test Pitch 比 S0 高 3.24%，不能忽略定位代价。U2 的 test XY=52.848，仍是总表最优，相对 S0 下降 2.28%；但它相对 S2 的“离散五项改善、连续八项退化”说明离散图像与连续序列的收益并不等价。

部分方向在 validation 与 test 之间发生变化：U1 相对 S2 的 Yaw 在 validation 下降 0.99%，test 却增加 0.25%；其离散 FID 则由 validation 更差变为 test 更好。P3 相对 S1 的 XY 和离散 LPIPS 在 validation 更好，test 略差。这些都是各自 split 内的对照，不把 validation 和 test 数值相减，也不把 FID 在不同样本集上的大小用于跨 split 排名。

**对论文贡献的当前判断。** P3 已超过 G0 的 validation／test 离散 PSNR（14.455 对 14.402；14.132 对 14.078），因此应删除原文“新组 PSNR 都低于单任务生成”的结论。但 G0 的离散 LPIPS 仍更好，J8 的离散 FID 仍优于各 Stage 2 模型；完整 pose 与生成也没有同一个候选在所有指标上占优。五个新组的 validation／test 四项定位宏平均均优于 L0，不过 J8 自身已经具备这一优势，且 Stage 2 多训练了 1800 步。S0 的 test XY 从 J8 的 55.512 降至 54.078，说明续训本身就有收益；辅助贡献应在此基础上说明，例如 U2 的 XY、U1 的连续生成、P3 的生成及部分朝向收益，并完整报告退化项。当前可以报告这些具体增益，尚不能声称全面的三级排序、全部增益均来自辅助 loss，或仅凭追加步数证明方法节省算力。

**下一步优先级（仅计划，2026-10-28 三组配方已修订，本次未启动实验）：**

1. 先补齐原定“先七组”中 B1、U3 的训练与评测证据。B1 联合 P3 的 perception 0.05 ramp 与 U2 的 aux-loc 1 ramp；在相同 validation local1200／1800 节点与 S0、P3、U2 形成四格比较，检验联合作用，不能从单项 loss 的结果预设收益相加。U3 补齐 aux-loc 终值 4 的既定权重端点，不改变数据划分或评测设置。
2. 基于 validation 保留明确分工：P3 为 perception 候选，U2 为 XY 候选，U1 为 XY／Z 与 Yaw 取舍候选，S2 为 Pitch 参照；S0 保持同更新预算的无辅助对照。P1／P2 继续作为取消 ramp 和增大权重的对照，不立即扩大 perception 网格。
3. 按已有设计补齐独立 U5（aux-loc 终值 0.5）：重点观察 validation XY 是否继续下降，以及 Pitch／Yaw 和生成指标是否出现进一步代价。把它作为权重下降方向的边界检验，不预设它优于 U2，也不因此继续新增 0.25 等更小权重实验。
4. 看完前七组的 validation 后，再按原计划人工决定 P4／U4：P4 只把 P3 的 attention weighting 改为 false，直接对照 P3；U4 只把 U2 的 candidate_tau 改为 0.001，直接对照 U2。候选配方与 checkpoint 以 validation 为依据，test 报告既定终点；保留完整定位、离散与连续指标，避免事后只选有利列。三图筛选完成后，再考虑将有明确任务收益的候选迁移至 Seen-10，而非直接把当前结果视为完整 benchmark 结论。


2026-10-14 当前执行命名：以下 2026-10-12／13 的历史分析中“A1～A3 用 J8”表示当时的研究建议；实际执行请使用独立 `A1_J8`／`A2_J8`／`A3_J8` 运行名。旧 A1～A3 保持 J4 默认，不能通过给旧名字加 `--joint-base J8` 覆盖旧实验目录。上述为当时命名约定；A1_J8 与 Stage 2 已于 2026-10-22 更新结果，见本表。

<a id="stage2-results-20261022"></a>

## 2026-10-22：J8 两阶段微调结果分析

**结论：续训本身已带来定位收益；辅助目标在部分指标上有增益，但尚未实现“辅助＋联合 > 联合 > 单任务”的全面排序。** 四组编号沿用配置中的 S0～S3，分别为无辅助、perception、aux-loc、both。下面先报告预先指定的 local1200 主比较，再用 local1800 作补充，不能只选最后节点替代主结果，也不把不同 checkpoint 的最好指标拼成一个模型。

以下仍使用上方结果表所链接 JSON 的 `metrics_macro_map`、三图等权、inference seed=42。J8 是 Stage 1 初始化参照；L0/G0 是各自单任务参照，训练预算与两阶段模型不同。

| 实验 | checkpoint | XY↓ | Z↓ | Pitch↓ | Yaw↓ | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 5850 | 48.459 | 2.878 | 2.818 | 26.235 | — | — | — | — | — |
| G0 | 5850 | — | — | — | — | 14.402 | **0.418** | **0.613** | 0.520 | <u>62.252</u> |
| J8 | 5850 | 47.113 | 2.665 | 2.548 | 25.070 | 14.222 | 0.416 | 0.623 | 0.517 | **61.569** |
| S0_J8 | 5850 + 1800 | 45.243 | 2.642 | 2.460 | <u>24.058</u> | 14.337 | 0.416 | 0.617 | **0.522** | 62.816 |
| S1_J8 | 5850 + 1800 | 45.275 | <u>2.598</u> | 2.473 | 24.254 | **14.376** | 0.417 | 0.616 | <u>0.521</u> | 63.214 |
| S2_J8 | 5850 + 1800 | <u>44.773</u> | 2.673 | **2.348** | **23.847** | <u>14.370</u> | 0.417 | <u>0.615</u> | <u>0.521</u> | 62.791 |
| S3_J8 | 5850 + 1800 | **44.358** | **2.591** | <u>2.431</u> | 24.088 | 14.353 | **0.418** | 0.617 | 0.520 | 62.596 |
<!-- | S0_J8 | 5850 + 1200 | 47.016 | 2.796 | 2.410 | 24.559 | 14.232 | 0.412 | 0.619 | 0.520 | 63.491 |
| S1_J8 | 5850 + 1200 | 47.389 | 2.672 | 2.483 | 24.755 | 14.348 | 0.419 | 0.619 | 0.520 | 62.466 |
| S2_J8 | 5850 + 1200 | 47.943 | 2.806 | 2.190 | 23.810 | 14.381 | 0.421 | 0.620 | 0.518 | 64.154 |
| S3_J8 | 5850 + 1200 | 48.228 | 2.732 | 2.509 | 24.649 | 14.336 | 0.418 | 0.618 | 0.518 | 64.313 | -->

**1200 主比较：没有辅助组全面优于同节点 S0。** S1 的 PSNR 增加 0.116、FID 降低 1.024，但 XY／Pitch／Yaw 均略差。S2 的 Pitch／Yaw 分别下降 9.11%／3.05%，且三图朝向误差均改善，是值得复核的方向；代价是 XY 增加 1.97%、FID 增加 0.664，LPIPS 与 Boundary F1 也略差。S3 的 XY 增加 2.58%、FID 增加 0.823，尚未显示两项辅助相加就优于单项辅助。

**1800 补充比较：S2 与 S3 分别偏向朝向／感知指标和空间定位，但收益不全面。** S2 相对 S0 的 XY／Pitch／Yaw 下降约 1.04%／4.52%／0.88%，LPIPS 下降 0.00118；Z、Boundary F1 变差，FID 仅降低 0.025，不能称为有实质改善。S3 的 XY／Z 分别下降 1.96%／1.96%，三图 XY 均改善（ancient／dust2／nuke 分别减少 1.809／0.513／0.332），但 Yaw 略差，Boundary F1 从 0.522353 降至 0.520301；LPIPS 仅下降 0.000114。S1 的 PSNR 在此节点最高，但 FID 比 S0 差 0.397，暂不优先扩展 perception 权重网格。

**必须扣除额外续训的作用。** S0@1800 自身相对 J8 的 XY 已下降 3.97%，定位四项宏平均均改善；S3 相对 J8 的 XY 下降 5.85%，其中相对同节点 S0 的增量只有 1.96%。所有 Stage 2 已评测节点的 FID 都差于 J8@5850；与 G0@5850 相比，所有节点的 PSNR、LPIPS、FID 仍未超过单任务生成。虽然四组定位四项宏平均均优于 L0，J8 本身已经有该优势，且两阶段额外训练消耗不能忽略。

**实际训练与归因边界。** 四组来源记录均指向同一 J8@5850；各组保存的 `config.json` 均记录实际加载 2830 个键，missing／unexpected／missing-trainable 均为空，例如 [S0 config](outputs/csgo_1b/exp32_3maps_stage2_control_j8/config.json)。1200／1800 模型的 safetensors 头部长度与文件大小吻合，并可见 optimizer、scheduler、RNG 和 Trainer 文件；本次没有反序列化恢复或重新运行模型。

| 实验 | GPU×microbatch×GAS | local1800 的 epoch | local1200 源样本引用数（推算） | local1800 源样本引用数（推算） |
|---|---|---:|---:|---:|
| S0_J8 | 2×64×1 | 15.3846 | 153,600 | 230,400 |
| S1_J8／S2_J8 | 2×8×8 | 15.2559 | 152,640 | 228,960 |
| S3_J8 | 2×4×16 | 15.2560 | 152,560 | 228,840 |

样本引用数根据已保存运行参数、15,000 条训练源样本和当前 [batch sampler](unilip/train/nonmix_trainer.py)／Trainer 的尾批行为推算，并非历史逐样本轨迹实测。S0 每 epoch 丢弃 24 条；S1/S2 的跨 rank 对齐补齐 8 次引用，epoch 尾更新只有 32 条；S3 尾更新只有 24 条。名义有效 batch 都为 128，但实际曝光差约 0.63%～0.68%，采样、梯度累积和随机轨迹也不同。因此这里只能称为同 optimizer 更新数的条件比较，不能把小幅差异完全归因于辅助 loss。已有 aux-loc 冻结检查也不能替代这批 S2/S3 的端到端梯度诊断；目前未据此发现足以判废这些结果的证据。

**下一步优先级（仅计划，本次不修改训练配置、不启动实验）：**

1. 先统一可承受的实际布局，优先在共同的双卡 microbatch=4、GAS=16 下复核 S0、S2、S3，沿用相同 J8 权重、LR、数据顺序和 1200／1800 节点。重点判断 S2 的朝向收益、S3 的 XY/Z 收益能否保留；先控制 batch／尾批混杂，再扩大辅助权重搜索。
2. 优先保留 S2（朝向与 LPIPS 候选）和 S3@1800（XY/Z 候选），暂不指定统一胜者。若匹配布局后仍损害边界或生成分布，再单因素试 αaux=1（当前为2），其余配方固定；不同时改权重、ramp 和 LR。
3. 用逐图和配对样本分析判断小差值是否稳定，并补少量独立 Stage 2 seed；这只验证固定 J8 初始化下的随机性，完整方法稳健性还需要独立 Stage 1 seed。记录实际 GPU 时间，不能根据追加 1800 steps 就声称总成本更低。
4. 基于 validation 固定候选及 checkpoint 后，再做离散 test 和连续序列评测，并将确认有效的配方迁移到 Seen-10。当前只有单 seed 的三图 validation 结果，不据此宣称显著提升或完整 benchmark 结论。

本次同步新增 8 个实验／checkpoint 组合、16 行已完成任务结果，600 节点保留缺项；S1 在 `exp31/` 下的副本与标准目录结果相同，未重复计数。训练、推理、评测命令均未执行。

<a id="j9-j10-results-20261013"></a>

## 2026-10-13：J9/J10 结果分析与下一步计划

当前建议继续以 **J8（αloc=8）作为辅助四格的共同配方**，保留 **J9（αloc=12）作为定位优先候选**，暂不扩大 αloc 扫描。J9 提高了 XY、Pitch、Yaw 精度，但牺牲 Z 和生成表现；J10（αloc=16）在两个已评测节点均出现定位退化。结果支持“主任务配重很重要”，尚不支持“同一联合 checkpoint 全面优于两个单任务模型”或“辅助 loss 已经有效”。

以下均为 `seen_validation`、推理 seed=42、三图等权宏平均、checkpoint-5850；L0 与 G0 是两个独立的单任务模型。来源为上方定位和生成结果表链接的完整 JSON。

| 实验 | XY↓ | Z↓ | Pitch↓ | Yaw↓ | PSNR↑ | SSIM↑ | LPIPS↓ | Boundary F1↑ | FID↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 定位单任务 | 48.459 | 2.878 | 2.818 | 26.235 | — | — | — | — | — |
| G0 生成单任务 | — | — | — | — | 14.402 | 0.418 | 0.613 | 0.520 | 62.252 |
| J8，αloc=8 | 47.113 | 2.665 | 2.548 | 25.070 | 14.222 | 0.416 | 0.623 | 0.517 | 61.569 |
| J9，αloc=12 | 44.067 | 2.907 | 2.421 | 24.334 | 14.087 | 0.416 | 0.638 | 0.501 | 64.164 |
| J10，αloc=16 | 82.350 | 4.247 | 2.769 | 36.257 | 14.498 | 0.421 | 0.612 | 0.521 | 62.497 |

J9 相对 J8 的 XY 降低 **6.47%**，Pitch／Yaw 降低 **4.97%／2.94%**，但 Z 增加 **9.08%**，表中五项生成宏平均全部变差；其中 LPIPS 增加 **2.28%**，Boundary F1 从 `0.516770` 降到 `0.501416`，FID 从 `61.569` 增到 `64.164`。SSIM 的差异在三位小数下被隐藏（`0.416095→0.415637`），结论按完整精度判断。相对 L0，J9 的 XY／Pitch／Yaw 改善 **9.06%／14.11%／7.25%**，Z 略差 **1.03%**；生成五项宏平均均差于 G0。因此 J9 是性能折中的另一候选，不能只因 XY 最低就替换共同基线。

逐图核对也显示收益与代价：J9 对 J8 的 XY 在 ancient／dust2／nuke 均改善（`53.272→47.784 / 50.999→50.272 / 37.069→34.145`），但三图 Z、LPIPS、Boundary F1、FID 均变差。J9 的三图 XY 均优于 L0，但 dust2 优势仅 `50.387→50.272`；dust2 的 Yaw `20.594` 仍差于 L0 的 `20.445`，nuke 的 Z `2.745` 仍差于 L0 的 `2.536`，不能宣称逐图全指标占优。

J10 相对 J8 的定位四项宏平均均退步，XY 增加 **74.79%**，三图 XY 都变差；其生成 PSNR／SSIM／LPIPS／Boundary F1 宏平均略优于 G0，FID 略差（`62.497` 对 `62.252`）。这些小差值来自单次训练与固定推理 seed，尚无显著性或等效性结论。J10 的生成恢复也不能抵消其定位退步。

完整权重扫描呈非单调结果，不能将此前 αloc=1→8 的趋势外推：

| αloc（实验） | XY@2400↓ | XY@5850↓ | LPIPS@5850↓ | FID@5850↓ |
|---|---:|---:|---:|---:|
| 1（J6） | 134.598 | 89.705 | 0.603 | 61.294 |
| 2（J3） | 112.408 | 67.537 | 0.609 | 65.062 |
| 4（J7） | 88.933 | 57.560 | 0.621 | 62.457 |
| 8（J8） | 67.344 | 47.113 | 0.623 | 61.569 |
| 12（J9） | 61.070 | 44.067 | 0.638 | 64.164 |
| 16（J10） | 130.298 | 82.350 | 0.612 | 62.497 |

αloc=1→12 时，两个节点的 XY 持续改善、5850 的 LPIPS 持续变差；αloc=16 时发生反转。J9@2400 的 Pitch=`4.047`，差于 J8 的 `2.504` 和 L0 的 `3.278`，到 5850 才改善。这进一步说明应比较完整定位分量和两个任务，并保留中后期评测，不能根据 2400 的 XY 单项决定最终方案。

本次还核对了训练证据：J9 的 [实际配置](outputs/csgo_1b/exp32_3maps_joint_constloc12/resolved_config.yaml) 与 J8 仅 αloc 不同，运行计划均为单 GPU／microbatch=64／GAS=2／seed=42／50 epoch；J8 计划记录从自身 3600 恢复，J9 计划无恢复记录。J10 的评测配置与 αloc=16 配方一致，但本地训练计划、Trainer 状态和模型仍待同步，不能用评测配置替代训练证据，也不能把退化直接解释为梯度冲突、爆炸或实现错误。

已有 Trainer 日志提供了一个待查线索：J8 的 [Trainer 状态](outputs/csgo_1b/exp32_3maps_joint_constloc8/trainer_state.json) 中 `grad_norm>1` 为 **232/585 个记录点（39.66%）**，J9 的 [Trainer 状态](outputs/csgo_1b/exp32_3maps_joint_constloc12/trainer_state.json) 为 **454/585（77.61%）**；step>3600 时分别为 **9/225（4.00%）**、**95/225（42.22%）**。这是每 10 步记录的总梯度范数，不能写成所有 optimizer steps 的实际裁剪比例，也不能单凭它推断哪一任务或哪一模块受到抑制。需进一步核对范数记录位置、裁剪阈值与参数范围；若使用全局裁剪，任务间可能通过共同裁剪系数发生耦合，但这仍是待验证解释，尤其尚无 J10 的对应日志。

后续按以下优先级推进；本次只更新结果及方案，未新增配置或启动实验：

1. **先补运行证据和有针对性的诊断。** checkpoint 到齐后核对 J10 与 J8/J9 的真实初始化、配方、有效 batch、恢复记录、LR 和训练曲线；同时核查 J8 的完整恢复轨迹。挑选已有代表性 checkpoint，在相同源样本和随机性下测累积完成、裁剪前的共享 LLM 与两类私有模块的未加权／加权任务梯度、共享梯度余弦及实际裁剪系数；必要时记录 Adam 后的模块更新量。先记录真实有效 batch 的行为，不将历史 12 个小 batch 的诊断外推为全程机制；暂不增加 α>16 或密集 α 网格。
2. **用 J8 推进既定辅助四格，优先 A1 perception-only，其次 A2 aux-loc-only，最后 A3 两者都有。** 主任务固定 αloc=8，A1 仍按 step600 开启权重 0.1，A2 仍按 `[0,1800]→[0,2]` 调度；先只改变组件开关。A0 仅在初始化、实际配置、训练布局、seed、步数和评测口径全部一致并注明来源时复用 J8；3600 主筛选需要 J8@3600 的匹配结果，缺失时先同步并补评，无法取得则补独立 A0，不能拿 J8@5850 比辅助组@3600。入围后统一比较 5850。各辅助组从原始预训练权重初始化；旧 J4 配方 A3 另建版本保留，不覆盖或中途换配方续训。
3. **按四格证据再单因素调辅助 loss。** 目标是保住 J8 的定位表现，同时缩小其相对 G0 的 LPIPS／PSNR／Boundary F1 差距；也同时检查 SSIM 与 FID。若 A2 损害生成，先单独降低 αaux；若曲线显示启用阶段产生扰动，再单独测试延迟启用。不要同时降低权重和延迟，也不立即在 J9 上复制整套四格。J9 保留为定位优先参照。固定 αloc=8 的裁剪阈值消融仅在梯度诊断支持、辅助四格仍不能解决问题时考虑，不与 loss 整体缩放同时展开网格。
4. **在最终研究结论前补稳健性对照。** 对 L0/G0、J8 和最佳辅助组补独立训练 seed，优先形成至少 3 个训练 seed 的匹配比较，推理 seed 固定以区分训练随机性；预算允许再纳入 J9。同时建议额外设置仅把定位单任务 αloc 从 1 改为 8 的独立控制，保留原 L0 正式结果，检查 J8 约 2.78% 的 XY 优势是否只是优化尺度差异。此控制不等价于匹配联合训练总梯度或裁剪轨迹，也不预设它会改善单任务。配方和 checkpoint 由 validation 选定后，再报告离散／连续 test，并迁移到 Seen-10 主实验。当前只匹配每任务数据曝光，不能声称同计算预算优于单任务。

方法参考应服从诊断结果：[GradNorm（ICML 2018）](https://proceedings.mlr.press/v80/chen18a.html)通过调节梯度尺度平衡任务，[PCGrad（NeurIPS 2020）](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)通过梯度投影处理任务干扰。前者可在确认尺度失衡后作为后备，后者需先确认稳定的共享梯度冲突；本项目当前证据不足以优先引入它们，更不能将预期收益写成实验结论。

2026-10-12 的 J5～J8 分析保留为当日历史判断；其中“J5/J8 元数据待同步”为当时状态，现状见本次实验记录表。

<a id="j5-j8-results-20261012"></a>

## 2026-10-12：J5～J8 结果分析与下一步计划

以下比较都使用 `seen_validation`、推理 seed=42、三图等权宏平均，并在同一 checkpoint 步数下进行；尚无多训练 seed 的复现，J5/J8 的实际训练布局仍待元数据同步核验。J6/J3/J7/J8 构成定位权重 αloc=1/2/4/8 的对照，除该权重外使用相同的 J3 配方：

| αloc（实验） | XY@2400↓ | XY@5850↓ | LPIPS@5850↓ | FID@5850↓ |
|---|---:|---:|---:|---:|
| 1（J6） | 134.598 | 89.705 | 0.603 | 61.294 |
| 2（J3） | 112.408 | 67.537 | 0.609 | 65.062 |
| 4（J7） | 88.933 | 57.560 | 0.621 | 62.457 |
| 8（J8） | 67.344 | 47.113 | 0.623 | 61.569 |

在这四个设置和两个已评测步数上，提高 αloc 持续改善定位 XY；5850 的生成 LPIPS 随之变差，FID 则非单调。J8@5850 的定位宏平均 XY/Z/Pitch/Yaw 为 `47.113/2.665/2.548/25.070`，四项均优于定位单任务 L0@5850 的 `48.459/2.878/2.818/26.235`。逐图仍有例外：dust2 的 XY 为 `50.999` 对 L0 `50.387`，Yaw 为 `20.761` 对 `20.445`。J8 的三图 LPIPS 均差于生成单任务 G0；FID 宏平均 `61.569` 优于 G0 的 `62.252`，收益主要来自 ancient（`57.782` 对 `61.463`），dust2 和 nuke 反而较差。因此目前只可说 J8 在此验证集的定位宏平均占优，不能宣称逐图或生成全指标胜出。逐图和完整精度见上方[定位结果](#定位-validation-结果)与[离散生成结果](#离散生成-validation-结果)的源 JSON。

J0/J1/J5/J3 的 2×2 对照显示定位头 LR 与 αloc 调度之间可能存在交互：S3 下，对齐定位私有 LR 后 XY@5850 从 J0 的 `82.517` 变成 J1 的 `88.997`（增加 `6.480`）；恒定 2 下，从 J5 的 `82.201` 变成 J3 的 `67.537`（降低 `14.664`）。这是单 seed 的配方现象，不能把 S3 的差异简单归因于终点权重较大或优化实现错误。生成端也需保留 J5 的 FID `59.700`、J6 的 PSNR `14.607` 与 LPIPS `0.603` 作为不同指标的对照，不能只凭定位排名选配方。

下一轮拟以 J8 配方作为统一 `JOINT_BASE`，保持现有 A1 perception、A2 aux-loc 的权重和启用时序，先评估 A1，再评估 A2，单组件作用明确后再评估 A3。`JOINT_BASE` 只复制配方，各辅助组仍从相同原始权重独立初始化，不加载 J8 的训练后 checkpoint。A0 仅在训练配置、初始化、seed、有效 batch、步数和评测口径全部核实一致后，明确注明来源并复用 J8 对应步数结果；不满足这些条件时单独训练 A0。A3 当前已有 J4 配方的计划记录；若统一改用 J8，须另建实验版本和输出目录，不能覆盖该计划或直接从旧 A3 checkpoint 换配方续训。

执行上，2400 只作观察，3600 是辅助组主要筛选点；若远端保留 J8@3600 checkpoint，可同步后补做同口径 validation，否则不跨步数比较 J8 与辅助组。入围候选和匹配对照继续在相同 5850 步比较三图定位与生成指标，不以单一 FID 波动淘汰或选优。随后对少数候选补独立训练 seed，冻结基于 validation 的配方／checkpoint 选择，再报告离散测试和连续测试，并将选定的联合及辅助配方迁移到 Seen-10 主实验复核。J5/J8 的本地训练状态与 checkpoint 到齐后还需核对配置及文件；指标表已按同步的评测 JSON 记录。本节为后续方案，尚未修改配置或启动新实验。

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
