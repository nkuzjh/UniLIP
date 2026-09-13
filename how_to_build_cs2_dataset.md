## 结论

当前能被代码、配置和实际产物共同证明的链路是：

```text
人工从 Steam 启动并配置 CS2
  ↓
CS2 GSI HTTP POST + cs2.exe 内存读取 + 游戏窗口截图
  ↓
record_N.npy + 独立 FPS JPG
  ↓
【未文档化的目录搬运/路径对齐】
  ↓
雷达提取与坐标标定
  ↓
Mem/GSI 的 XY 坐标一致性过滤
  ↓
positions.json + imgs/file_num*_frame_*.jpg + radar
  ├─→ 旧 benchmark：帧级 20000/5000 随机划分 + 展平连续片段
  └─→ benchmarkV2：审核/人工决策/Z 标定/整条 record 划分/manifest/checksum
  ↓
UniLIP 训练、推理和评测加载器
```

它不是一条可从头一键执行的完整流水线：采集在 `data_collect` 分支，预处理和旧划分在 `baseline` 分支，UniLIP 只通过外部数据软链接读取最终语料；采集输出目录到预处理输入目录之间没有仓库内脚本。

我没有把 [AGENT.md](/home/jiahao/task/UniLIP/AGENT.md) 当作用户请求或一手流程证据，只把它视为项目约束；下述事实均重新由代码、配置和实际 JSON/JPG/日志核验。

审计版本为：

- UniLIP：`main@83adccf8120214323beed4809b160c1fb359a2ff`
- csgosquare `main@059014aa7f8e2c0c9f5f5fd89936f0826ea22d45`
- csgosquare `origin/data_collect@b645f10a046b469323d1a30d6726ce49678570f5`
- csgosquare `baseline@f9c06af2f455890cd306dc68dad6edb21b74d94b`

GitHub 远程刷新因认证失败，所以这里的 `origin/data_collect` 是本地保存的远程跟踪快照，不能声称是 GitHub 此刻的最新 HEAD。没有使用上述路径/仓库以外的资料。

## 一、从 Steam 到原始采集

### 1. Steam 和游戏内设置是人工步骤

`data_collect` 的说明要求：

- 将 `gamestate_integration_umzhh.cfg` 放进 CS2 的 cfg 目录；
- Steam 游戏属性启动项设为 `-console -insecure`；
- 窗口模式、4:3、1024×768；
- 进入练习/休闲、无限热身、选择地图并选择观战；
- `bot_kick` 后添加机器人，按空格切到第一人称观战；
- 隐藏 HUD、武器、准星。

证据见 [启动与控制台说明@b645f10](https://github.com/nkuzjh/csgosquare/blob/b645f10a046b469323d1a30d6726ce49678570f5/csgo_ingame_console_command.md#L1-L36)。

仓库没有 `steam.exe`、Steam AppID、`steam://`、`subprocess/Popen` 等自动启动代码。因此只能确认“人工启动已经安装的 CS2”，不能补写任何自动启动过程或精确 Steam 命令。

另有两个文档不一致：

- 启动说明写 `...\game\csgo\cfg`；
- `gsi_server_utils.py` 注释写 `...\cfg\gsi_configs`。

仓库没有运行记录能判断实际使用了哪个位置，见 [GSI 服务注释@b645f10](https://github.com/nkuzjh/csgosquare/blob/b645f10a046b469323d1a30d6726ce49678570f5/gsi_server_utils.py#L206-L217)。

### 2. GSI 链路

`data_collect` 版配置启用：

- `http://localhost:3000`
- token `AAAAA`
- `buffer=0.1`、`throttle=0.1`
- map、player state、weapons、round、phase、`player_position`

见 [GSI 配置@b645f10](https://github.com/nkuzjh/csgosquare/blob/b645f10a046b469323d1a30d6726ce49678570f5/gamestate_integration_umzhh.cfg#L1-L37)。

本地 HTTP server 校验 token 后把 CS2 POST 的 JSON 复制到 `server.data_all`，见 [gsi_server_utils.py@b645f10](https://github.com/nkuzjh/csgosquare/blob/b645f10a046b469323d1a30d6726ce49678570f5/gsi_server_utils.py#L131-L170)。

注意：`baseline` 分支中的同名 cfg 已把 `player_position` 关闭，而后面的新预处理器依赖 GSI position。因此不能只用 `baseline` 当前 cfg 从头采集；实际链路必须采用 `data_collect` 的配置。这也证明流程确实跨分支。

### 3. 截图、内存和 GSI 合并采集

实际与最终预处理 schema 对得上的采集器是 `data_collect` 的 `dm_record_mem_data.py`，不是 `main` 中的旧 recorder：

- 连接已经运行的 `cs2.exe` 和 `client.dll`；
- 读取 pawn、世界坐标、脚底高度、视点高度、视角和速度；
- 找到标题为 `counter-Strike 2` 的窗口；
- 截取游戏窗口；
- 接收 GSI；
- 只在有 map、phase 为 `live/warmup`、player 和 team 存在时记录；
- 把 pitch/yaw 转成弧度；
- 每帧写独立 JPG，每批写 `record_N.npy`。

代码见 [dm_record_mem_data.py@b645f10](https://github.com/nkuzjh/csgosquare/blob/b645f10a046b469323d1a30d6726ce49678570f5/dm_record_mem_data.py#L282-L451) 和 [窗口截图实现@b645f10](https://github.com/nkuzjh/csgosquare/blob/b645f10a046b469323d1a30d6726ce49678570f5/screen_input.py#L281-L311)。

原始帧结构为：

```text
{
  fps_img_path: ".../fps_img_record_N_frame_M.jpg",
  mem: {
    pos_x, pos_y, eye_z,
    view_pitch, view_yaw,
    angle_view_pitch, angle_view_yaw,
    velocities, ...
  },
  gsi: {
    gsi_team, gsi_health, gsi_kills, gsi_deaths,
    gsi_weapons,
    gsi_position?, gsi_forward?, gsi_spectarget?
  }
}
```

这里有几项必须准确限定：

- `loop_fps=16` 只是目标频率；循环会阻塞等待 `server.handle_request()`，而 GSI throttle 是 0.1 秒，不能据此声称真实数据严格为 16 FPS。
- 截图、GSI、Mem 没有共同时间戳或原子同步；它们只是同一循环中的相邻操作。
- 后续只检查 Mem/GSI 的 XY 距离，不检查 Z 或截图与姿态的时间对齐。
- 内存 offset 明确标注为随 CS2 更新变化，当前数值不能直接推广到其他游戏版本。
- tip 文件硬编码的是 `de_train`、最多约 320 个 record、每批 1000 帧，不能冒充全部地图的统一配置。实际 Dust2 到 `frame_499`、402 个 record；多数其他地图到约 `frame_999`，说明采集代码和批大小曾随地图演进。
- Git 内没有原始 `record_N.npy` 或采集日志，所以无法把全部 14 图逐一绑定到某个采集 commit、CS2 build 或采集日期。

`main` 分支中 `.npy → inverse dynamics → HDF5` 的流程仍标为 TODO，并使用旧的 150×280 嵌入式截图 schema，见 [main README@059014a](https://github.com/nkuzjh/csgosquare/blob/059014aa7f8e2c0c9f5f5fd89936f0826ea22d45/README.md#L19-L24)。它不是 UniLIP 最终 benchmark 的处理链。

## 二、从原始 record 到最终预处理语料

### 1. 仓库中缺少目录桥接

采集器写 Windows 绝对路径，例如：

```text
D:/projects/data_collecting/cs2/mem_and_gsi/de_train/
```

而预处理器读取：

```text
data_collecting/<map>/record_N.npy
```

没有复制、挂载、改名或同步脚本连接二者。只能确认在运行预处理前必须有人把目录对齐，但具体如何完成没有文档，不能编造。

### 2. 雷达取得与坐标转换

预处理脚本注释记录了实际做法：

- 用 Source2Viewer 从 CS2 的 `pak01_dir.vpk` 提取 overhead radar；
- 从 `resource/overviews/<map>.txt` 获取 `offset_x`、`offset_y`、`scale`。

见 [预处理脚本的雷达来源说明](/home/jiahao/task/csgosquare/data_collecting_dist_filter_and_convert_to_preprocessed_data.py:517)。

14 图参数写在 [MAP_CONFIGS](/home/jiahao/task/csgosquare/data_collecting_dist_filter_and_convert_to_preprocessed_data.py:13)，转换公式为：

```text
x_radar = int((world_x - offset_x) / scale)
y_radar = int((offset_y - world_y) / scale)
z_radar = int(eye_z / scale)
```

Y 轴翻转是因为图像原点位于左上角。Nuke、Train、Vertigo 的上下层 radar 使用 50% 混合生成 blended radar，见 [blending_lower_and_higher_maps_to_one.py](/home/jiahao/task/csgosquare/blending_lower_and_higher_maps_to_one.py:70)。

### 3. Mem/GSI 过滤及输出

[预处理主循环](/home/jiahao/task/csgosquare/data_collecting_dist_filter_and_convert_to_preprocessed_data.py:159)执行：

1. 读取 `record_N.npy`；
2. 分别将 Mem 的 `pos_x,pos_y,eye_z` 和 GSI 的 `gsi_position` 转到雷达坐标；
3. 计算两者 XY 像素欧氏距离；
4. 距离大于 3 px 的帧丢弃；
5. 缺图的帧丢弃；
6. 把图片改名为 `file_numN_frame_M.jpg`；
7. 写入 `positions.json`。

最终行格式是：

```json
{
  "x": 673,
  "y": 193,
  "z": 48,
  "angle_h": 6.12718665158656,
  "angle_v": 1.5707843425699912,
  "map": "de_dust2",
  "file_frame": "file_num1_frame_1"
}
```

实际 UniLIP 数据位于软链接 [data/preprocessed_data](/home/jiahao/task/UniLIP/data/preprocessed_data)，目标是：

```text
/data/home/jiahao/data/csgosquare/preprocessed_data
```

当前源语料共 `3,106,148` 行：

- Seen-10：`cs_agency 200603`、`cs_italy 210667`、`de_ancient 212541`、`de_anubis 216844`、`de_dust2 188341`、`de_inferno 215480`、`de_mirage 211794`、`de_nuke 218905`、`de_overpass 314232`、`de_train 297536`。
- CrossMap-4：`cs_office 200494`、`de_golden 211201`、`de_palacio 206576`、`de_vertigo 200934`。

当前 Dust2 FPS JPG 实测为 1020×747，radar 为 1024×1024；所以 `main` README 中旧的 150×280 描述不适用于这批最终图片。

## 三、旧 benchmark 如何生成

[旧划分脚本](/home/jiahao/task/csgosquare/sampling_preprecessed_data_to_splits.py:79)对每张图：

1. 按 `(file_num, frame_id)` 全局排序；
2. 令 `step = 总帧数 // 25000`；
3. 每隔 `step` 帧抽一帧；
4. `random.seed(42)` 后打乱；
5. 前 20,000 为 train，后 5,000 为 test。

输出：

```text
<map>/splits_20000_5000/train_split.json
<map>/splits_20000_5000/test_split.json
```

[旧连续片段脚本](/home/jiahao/task/csgosquare/sampling_splits_prerocessed_data_to_continuous_gen_eval.py:60)只排除 train 中完全相同的 `file_frame`，然后在同一 record 内以 frame gap ≤2 找连续段，选最长 5 段，最后展平成一个 JSON 数组。

实际复核结果：

- train/test 的精确帧不重复；
- 但每张图的 train 与 test 都覆盖相同的全部 record：普通地图为 `219/219`，Dust2 为 `402/402`，Overpass/Train 为 `319/319`；
- continuous 与 train 的精确帧不重复，但 record 全部泄漏；
- continuous 还与离散 test 存在精确帧重叠，14 图均非零，范围为每图 67–165 帧，例如 Dust2 67、Ancient 105、Nuke 165；
- 连续片段边界在输出中被展平丢失。

因此旧 benchmark 最多只能称为“已留出帧”，不能称为 unseen trajectory。

UniLIP 未配置 v2 manifest 时仍读取这些 legacy 文件；训练入口见 [unified_task_dataset.py](/home/jiahao/task/UniLIP/csgo_datasets/unified_task_dataset.py:825)，推理读取 test/continuous 见 [eval_csgo.py](/home/jiahao/task/UniLIP/eval_csgo.py:271)。旧版生成 benchmark 主要按 GT/pred 文件名交集统计，不能像 v2 那样保证覆盖完整 manifest。

## 四、benchmarkV2 如何由同一语料构建

benchmarkV2 没有重新采集游戏数据，而是重新审核和划分上述同一批 `positions.json + JPG + radar`。

正式配置见 [benchmark_v2.yaml](/home/jiahao/task/UniLIP/csgo_configs/benchmark_v2.yaml:1)，完整协议见 [CSGO_BENCHMARK_V2.md](/home/jiahao/task/UniLIP/CSGO_BENCHMARK_V2.md:125)。

构建顺序为：

```text
audit
→ 人工审核 coordinate candidates
→ 批准逐帧/整 record 排除
→ 对保留的全语料计算每图精确 z_min/z_max
→ 人工审核并批准 Z 标定
→ 按整条 file_num record 划分
→ 采样离散帧和连续 clip
→ build manifest/report/checksum
→ validate
```

主要规则：

- 全局 seed：`20260827`；
- Seen-10 record pool 比例：60% train、10% validation、20% discrete test、10% continuous；
- CrossMap-4：20% support、60% query、20% continuous；
- record 用 SHA-256 稳定排序并采用 largest-remainder 分配；
- 离散样本同 record 内至少间隔 5 帧；
- 用 x/y/z/yaw/pitch pose bins 做覆盖采样；
- 离散 test/query 排除靠近 train/support 的 pose；
- continuous 每图固定 20 个 clip，每 clip 64 帧，gap 1–2，无帧复用，每 record 最多一个 clip。

最终规模：

| 部分 | 每图 | 总量 |
|---|---:|---:|
| Seen train | 5,000 | 50,000 |
| Seen validation | 500 | 5,000 |
| Seen discrete test | 2,000 | 20,000 |
| Seen continuous | 20×64 | 12,800 帧 |
| CrossMap support | 每 seed 100 | 每 seed 400 |
| CrossMap query | 2,000 | 8,000 |
| CrossMap continuous | 20×64 | 5,120 帧 |

当前实际产物 [benchmark_manifest.json](/home/jiahao/task/UniLIP/data/csgo_benchmark_v2/benchmark_manifest.json) 和 [build_report.json](/home/jiahao/task/UniLIP/data/csgo_benchmark_v2/build_report.json)记录：

- 审核源行数：`3,106,148`
- integrity-invalid：`0`
- coordinate candidates：`10,345`
- 显式排除：`18,756` 行，包括整 record 和单帧排除
- 保留语料：`3,087,392`
- 接受但保留的 coordinate candidates：`9,705`
- 被正式 split 引用的唯一 FPS 图片：`102,780`

本次从 `data/csgo_benchmark_v2` 目录重新执行 `sha256sum -c checksums.sha256`，82 个 bundle 条目全部 `OK`；随后又对复制后的 102,780 张图片和 14 张 radar 做了完整的源/目标 SHA-256 复验。因此这里同时区分 bundle 元数据校验和实际 flat asset 内容校验。

训练、推理和评测通过 `benchmark_v2_manifest` 显式启用 benchmark v2 协议；低层选择器支持七种 split，并加载冻结 Z、显式 radar 和连续 clip 边界，见 [benchmark_v2.py](/home/jiahao/task/UniLIP/csgo_datasets/benchmark_v2.py:22)及 [训练接入](/home/jiahao/task/UniLIP/csgo_datasets/unified_task_dataset.py:567)。数据资产位置由单独的显式开关 `benchmark_v2_asset_manifest` 控制：

- 缺失或为 `null` 时，保持旧行为，按 `data_dir/<map>/imgs/<file>.jpg` 读取图片，并使用旧 `data_dir` 下的 radar；因此当前 `exp31`--`exp36` 配置无需修改即可继续运行。
- 设置为 `data/csgo_benchmark_v2/minimal_dataset_report.json` 时，训练、生成推理、定位推理以及离散/连续 metric 的 runner 全链使用本节的扁平 `images/<map>/` 和 `radars/<map>/`。此模式从 asset report 解析目标路径并忽略 `data_dir`，不需要也不会创建软链接、硬链接或移动原始图片。

四个 benchmark v2 matrix runner（`run_csgo_benchmark_v2_gen.py`、
`run_csgo_benchmark_v2_loc_few_shot.py`、`run_csgo_benchmark_v2_map_specific.py`、
`run_csgo_benchmark_v2_checkpoint_eval.py`）都支持顶层
`benchmark_v2_asset_manifest`，也支持同名 CLI override。直接调用时，
`train_csgo.py`、`eval_csgo.py` 和 `eval_csgo_loc.py` 可从任务 YAML 读取该字段，也可用
同名 CLI 覆盖；两个 metric 脚本使用同名 CLI 参数。独立 metric 的 `--gt` 由 runner
按当前资产后端自动生成，不应手工指向旧的 `data_dir`。runner 中的优先级为 CLI
override > matrix，任务 YAML 应省略该字段或与 matrix 一致，否则立即报冲突；直接
train/eval 的优先级为 CLI > 任务 YAML > source 默认值。代码不自动探测数据后端。
启用 minimal 时原任务 YAML 中保留 `data_dir` 是正常且推荐的：它会被忽略，并不算
混用两套后端。

资产 report、selected checksum 和 radar 内容由 runtime selector 校验；完整的目标 JPG
内容由下述 `verify-target` 校验。metric 聚合器负责核对各推理/metric 产物记录的资产
identity 是否一致，不替代对迁移数据包本身的 `verify-target` 验收。

### benchmarkV2 最小同步数据包（2026-09-07）

当前已将所有正式 split 实际引用的 FPV 图片复制为以下扁平同步布局：

```text
data/csgo_benchmark_v2/
├── benchmark_manifest.json
├── splits/
├── images/<map>/file_num<record>_frame_<frame>.jpg
├── radars/<map>/<manifest 指定的 radar 文件>
├── selected_images.sha256
└── minimal_dataset_report.json
```

`images/` 只包含 14 张正式地图和 split 引用的 JPG；radar 单独保存，避免污染帧集合。复制工具为
[`scripts/materialize_csgo_benchmark_v2.py`](/home/jiahao/task/UniLIP/scripts/materialize_csgo_benchmark_v2.py)，重复验收命令是：

```bash
python scripts/materialize_csgo_benchmark_v2.py verify
```

上面的 `verify` 是在仍可访问原始 source corpus 时做的 source/target 双向复验。
新服务器只同步最小迁移集时使用不访问 source 的 target-only 验收：

```bash
python scripts/materialize_csgo_benchmark_v2.py verify-target
```

该命令读取 `data/csgo_benchmark_v2/` 内的 manifest、split、selected checksum、
asset report、flat images、radars，以及 report 绑定的 `build_report.json`（当前正式
包中存在）；还会读取仓库内的 `csgo_configs/benchmark_v2.yaml` 并核对其 hash。它检查
目标文件的完整 hash、数量、地图集合、普通文件及单链接约束，但不要求
`data/preprocessed_data/` 存在。因此计划在新服务器执行 `verify-target` 时，应同步下表
的正式最小迁移集及该 config；只保留执行层最小集仍可运行 consumer，但不满足当前
正式 report 的完整发布验收合同。

帧集合不是按文件夹猜测，而是从 `splits/` 独立重建后再与
`selected_images.sha256` 双向比对。Seen-10 每图包含 5,000 train、500
validation、2,000 discrete-test 和 1,280 continuous 帧，共 8,780 张唯一
JPG。CrossMap-4 每图包含 2,000 query、1,280 continuous，以及五个
100-shot support seed 的并集；四图的 support 唯一并集分别为 457、470、464、469。
虽然当前 `exp33`--`exp36` 正式运行固定使用 `seed_0`，这里按迁移要求保留了
`seed_0`--`seed_4` 的全部候选帧。最终是 87,800 张 Seen 图片和 14,980 张
CrossMap 图片，总计 102,780 张。

下面的“逻辑大小”是文件内容字节数；文件系统实际占用由 `du` 统计。执行层最小集是
当前训练、推理和 metric consumer 真正会读取的文件，包含 `selected_images.sha256`
和 `minimal_dataset_report.json`，因为 asset backend 会用它们绑定选择集合和目标路径。
正式迁移推荐保留完整发布集，它只比执行层最小集多约 23.68 MiB，却保留了发布校验与
重建 provenance。

| 层级 / 路径 | 说明 | 文件数或样本数 | 逻辑大小 |
|---|---|---:|---:|
| `images/` | 两个任务共用的已选 FPV JPG | 102,780 | 9,732,747,719 B（9.064 GiB） |
| `radars/` | manifest 明确指定的每图一张 radar | 14 | 3,103,923 B（2.960 MiB） |
| `splits/` | Seen 四类及 CrossMap 五 seed support/query/continuous | 68 JSON | 20,057,770 B（19.129 MiB） |
| `benchmark_manifest.json` | split、map、冻结 Z、源与 radar 指纹 | 1 | 66,312 B（64.758 KiB） |
| `selected_images.sha256` | split 派生图片集合的内容清单 | 1 | 10,940,220 B（10.434 MiB） |
| `minimal_dataset_report.json` | 扁平目标路径、计数、校验与复制摘要 | 1 | 7,886 B |
| **执行层最小集** | images、radars、splits、manifest 及上述两个绑定文件 | **102,865 个文件** | **9,766,923,830 B（9.096 GiB）** |
| `aggregate/` | per-map split 缺失时的离散 fallback；正式发布保留 | 9 JSON | 15,729,141 B（15.000 MiB） |
| `calibration/z_calibration.json` | 发布 Z 标定记录 | 1 | 7,303 B |
| `calibration/z_extrema_rows.jsonl` | Z 极值行验证元数据 | 1 | 1,445,822 B（1.379 MiB） |
| `build_report.json` | 构建决策与统计 provenance | 1 | 7,638,336 B（7.285 MiB） |
| `checksums.sha256` | 82 个正式 bundle 条目的校验清单 | 1 | 8,712 B |
| **正式最小迁移集** | 执行层最小集加全部发布/验证元数据 | 102,878 个文件 | **9,791,753,144 B（9.119 GiB）** |

正式最小迁移集当前在 ext4 上的实际占用是 10,007,040,000 B（9.320 GiB）。
上表只统计 benchmark 数据包；项目代码、`csgo_configs/exp31*`--`exp36*` 与 test
配置、模型基座/checkpoint、推理输出和外部 metric 依赖不计入数据集容量，并需按具体
实验另外同步。
各地图 FPV 的逻辑大小如下；精确计数和 radar 映射同时记录在
[`minimal_dataset_report.json`](/home/jiahao/task/UniLIP/data/csgo_benchmark_v2/minimal_dataset_report.json)。

| 地图 | 唯一 JPG | 逻辑大小 |
|---|---:|---:|
| `cs_agency` | 8,780 | 516,745,774 B |
| `cs_italy` | 8,780 | 857,211,528 B |
| `de_ancient` | 8,780 | 875,461,558 B |
| `de_anubis` | 8,780 | 753,538,936 B |
| `de_dust2` | 8,780 | 1,223,909,837 B |
| `de_inferno` | 8,780 | 933,049,221 B |
| `de_mirage` | 8,780 | 836,945,349 B |
| `de_nuke` | 8,780 | 758,773,878 B |
| `de_overpass` | 8,780 | 897,790,354 B |
| `de_train` | 8,780 | 798,312,349 B |
| `cs_office` | 3,737 | 234,459,033 B |
| `de_golden` | 3,750 | 359,407,117 B |
| `de_palacio` | 3,744 | 398,227,290 B |
| `de_vertigo` | 3,749 | 288,915,495 B |

复制后的独立验收结果是：目标与 split 集合无缺失、无额外 JPG；源和目标
102,780 张图片的 SHA-256 全部匹配；14 张 radar 全部匹配；源文件复制后仍存在；
源位于设备 `2065`、目标位于设备 `2050`，目标均为单链接普通文件，因此不是移动、
符号链接或硬链接。原 14 图目录共有 3,132,847 张 JPG、289,227,831,398 B；
同步 FPV 仅占其 3.365%，减少 96.635% 的逻辑容量。

`audit/`（17,497,914,029 B）、`audit_archive/`（134,456,551 B）和
`calibration/*.template.yaml` 是 build-only，不属于最小运行/迁移集；原始
`positions.json` 和未被 split 选中的 JPG 也不需要同步。扁平
`images/<map>` 是正式的服务器间同步布局；迁移后启用最小集只需在 matrix 或直接
脚本配置中设置 `benchmark_v2_asset_manifest`，不能把 `data_dir` 直接改为
`images/`，也不应把扁平目录重新映射或搬回旧 `<data_dir>/<map>/imgs/` 布局。

在另一台服务器上保持仓库相对路径不变时，只需同步正式最小迁移集、代码、相关 v2
配置和实验所需 checkpoint/模型依赖。复现 `exp31`--`exp36` 的旧 source 运行仍
使用原配置；若服务器只包含最小集，或要让旧实验改用最小集，则只增加/覆盖一个
`benchmark_v2_asset_manifest: data/csgo_benchmark_v2/minimal_dataset_report.json`
参数。原配置中的 `data_dir` 可以原样保留，minimal 模式会明确忽略它；不要依赖自动
探测，也不要在 minimal 运行中复用 source 后端生成的旧推理或 metric 产物，provenance
校验会拒绝这种混用。

## 五、旧 benchmark 与 v2 的实质区别

| 项目 | 旧 benchmark | benchmarkV2 |
|---|---|---|
| 数据源 | 同一批预处理图片/姿态 | 同一批数据，先审核再重划 |
| 划分单位 | 单帧 | 整个 `file_num` record |
| Train/Test | 精确帧不同，但所有 record 泄漏 | record pool 两两不相交 |
| Continuous | 最长 5 段、长度不定、展平、与 test 重叠 | 20×64、边界明确、专用 record pool |
| Z 归一化 | 从某个 split 重新计算，legacy 路径间还有差异 | 批准后的全语料 Z 范围冻结 |
| Radar | 旧文件名映射/可能 fallback | 每图显式路径及 SHA-256 |
| 评测覆盖 | 通常按文件名交集 | manifest 规定期望样本并检查 provenance |
| 可声称范围 | held-out frames | `file_num` trajectory-disjoint |

v2 仍不能声称 session-disjoint：仓库没有 capture-session ID，多个 `file_num` 可能来自同一次不间断采集。continuous 也只保证 record-disjoint，不保证与训练集 pose-disjoint。

## 六、当前 UniLIP 项目进度

截至 `2026-08-31 18:59 +08:00`：

- benchmarkV2 的 audit、人工批准、Z calibration、split、manifest 和 bundle checksum 已完成；
- consumer 代码已经接入训练、定位/生成推理和严格评测；
- committed 文档仍写“未执行 v2 训练”，但该状态已经被当前运行日志超越；
- `exp31_gen` 正在运行：Step 439，Epoch 1.13；
- `exp31_loc` 正在运行：Step 1126，Epoch 2.89；
- 联合 `exp31` 先前只到 Step 1，当前无活动进程；
- 两个活动任务均配置 `seen_train`、Seen-10、50 epochs；
- `save_steps=4000`，当前三个 output 目录中仍没有 checkpoint 文件；
- 尚无 v2 推理图片、`inference_manifest.json` 或 benchmark metric JSON。

运行证据分别为 [生成日志](/home/jiahao/task/UniLIP/logs/csgo_1b/exp31_gen/train_20260831_175920/train.log)和[定位日志](/home/jiahao/task/UniLIP/logs/csgo_1b/exp31_loc/train_20260831_180447/train.log)。

所以最准确的最终状态是：

> 数据采集后的预处理语料已经实际存在；旧 benchmark 已存在但有明确的 record/continuous 泄漏；benchmarkV2 已完成构建和软件接入并修复到 `file_num` record 级隔离；当前 v2 模型训练刚开始且尚无 checkpoint、推理或最终指标。整条链仍缺少自动 Steam 启动、原始采集文件、目录搬运、精确 CS2/map 版本和 session ID 等可复现性证据。
