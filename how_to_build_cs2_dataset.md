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

本次从 `data/csgo_benchmark_v2` 目录重新执行 `sha256sum -c checksums.sha256`，82 个 bundle 条目全部 `OK`。这验证了生成的 JSON、manifest、report、calibration 和 checksum 文件；本次没有重新哈希外部软链接中的全部 102,780 张源图片，因此不把 bundle 校验夸大为完整外部图片复验。

训练和推理通过 `benchmark_v2_manifest` 显式启用，否则继续走 legacy。低层选择器支持七种 split，并加载冻结 Z、显式 radar 和连续 clip 边界，见 [benchmark_v2.py](/home/jiahao/task/UniLIP/csgo_datasets/benchmark_v2.py:22)及 [训练接入](/home/jiahao/task/UniLIP/csgo_datasets/unified_task_dataset.py:567)。

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