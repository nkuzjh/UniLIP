你负责将当前已 clone 的独立模型项目接入 CSGO Benchmark v2 Seen-10。

【项目参数】
MODEL_NAME=<X-VLA / RDT / OpenVLA-OFT / pi0.5 / OmniGen / ControlAR / Lumina-Image-2.0 / Show-o2-1.5B / Puffin / Janus-Pro-1B>
MODEL_TYPE=<VLA 或 GENERATION>
PROJECT_ROOT=<当前独立项目根目录>
UNILIP_ROOT=/home/jiahao/task/UniLIP
DATA_ROOT=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
BUILD_SHARED_EVALUATOR=<首个独立项目填1，其余填0>
SHARED_EVAL_DIR=/home/jiahao/task/csgo_benchmark_v2_eval_general
RUN_FULL=<准备完成后立即运行完整实验填1；仅完成代码、配置和smoke填0>
TRAIN_SEEDS="0"  # 只需要单种子结果即可

Table 1 的任务分工如下：

- VLA：X-VLA、RDT、OpenVLA-OFT、pi0.5，只接入 localization。
- GENERATION：OmniGen、ControlAR、Lumina-Image-2.0、Show-o2-1.5B、Puffin、Janus-Pro-1B，接入 discrete generation 和 continuous generation。
- 不要为了补齐三列而给模型实现其在 Table 1 中不承担的任务。这里的“三个任务”是 Table 1 整体的 localization、discrete generation、continuous generation。

目标不是做大规模框架重构或公平性审计，而是以最小改动打通当前模型在 Seen-10 上的训练、推理和统一评测，并得到可填入 Table 1 的结果。不要停留在方案阶段，也不要反复请求审批。

一、阅读与方案

1. 完整阅读当前项目的 README、环境文件、训练入口、推理入口、dataset/dataloader、processor、模型构造、checkpoint 保存/加载和配置系统。
2. 找出最适合复用的原生训练与推理路径，优先保留项目原有 Trainer、optimizer、distributed launcher、LoRA/finetune 方式和 checkpoint 格式。
3. 阅读以下 UniLIP 参考文件：
   - ${UNILIP_ROOT}/AGENT.md
   - ${UNILIP_ROOT}/CSGO_BENCHMARK_V2_NEW_SERVER_MIGRATION_CHECKLIST.md
   - ${UNILIP_ROOT}/CSGO_BENCHMARK_METRICS_ZH.md
   - ${UNILIP_ROOT}/benchmark_csgo_v1.py
   - ${UNILIP_ROOT}/benchmark_csgo_v1_conti.py
   - ${UNILIP_ROOT}/eval_csgo_loc.py
   - ${UNILIP_ROOT}/scripts/aggregate_csgo_benchmark_v2_metrics.py
   - ${UNILIP_ROOT}/csgo_configs/benchmark_v2.yaml
4. 先简要列出需要修改/新增的文件、数据接入点、模型输入输出适配方式和运行命令，给出变更方案并保存在当前项目根目录下。
5. 然后根据变更方案直接实施，不等待确认。
5. 只改当前独立项目以及 SHARED_EVAL_DIR；不要修改、移动或复制 DATA_ROOT 中的数据。

二、Seen-10 数据合同

必须由以下文件驱动数据读取，不能扫描 images/ 后自行重新划分：

- ${DATA_ROOT}/minimal_dataset_report.json
- ${DATA_ROOT}/benchmark_manifest.json
- ${DATA_ROOT}/splits/
- ${DATA_ROOT}/images/<map>/<frame>.jpg
- ${DATA_ROOT}/radars/
- ${DATA_ROOT}/calibration/

Seen-10 地图固定顺序：

cs_agency
cs_italy
de_ancient
de_anubis
de_dust2
de_inferno
de_mirage
de_nuke
de_overpass
de_train

数据量：

- seen_train：每地图5,000，共50,000。
- seen_validation：每地图500，共5,000。
- seen_discrete_test：每地图2,000，共20,000。
- seen_continuous：每地图20个clip，每个64帧；共200个clip、12,800帧。

使用 benchmark manifest/split row 给出的 sample ID、图像和 radar 映射。不要把 DATA_ROOT/images 直接当成旧式 data_dir，也不要依赖
<data_dir>/<map>/imgs/<frame>.jpg。

三、任务适配

如果 MODEL_TYPE=VLA：

1. 输入为当前第一视角图像、对应 radar/map 图像以及固定任务 instruction/map name。
2. 机器人 proprio/state 不适用于该任务，设为 zero/masked，禁止填入 GT pose。
3. 将动作输出以最小改动替换或适配为 horizon=1 的绝对5DoF：
   [x, y, z, pitch, yaw]。
4. 标签和预测统一使用 Benchmark v2 的归一化定义；Z 范围必须读取发布 calibration/manifest，不得从 test 重新估计。
5. 尽量复用原模型的视觉、语言和 action head；仅新增必要的双图输入适配、5D head或投影层。
6. 使用 seen_train 训练、seen_validation 选 checkpoint、seen_discrete_test 推理。
7. 输出：
        outputs/csgo_benchmark_v2_seen10/${MODEL_NAME}/seed_<seed>/localization/predictions.jsonl
   每行至少包含：
        sample_id、map_name、pred_x、pred_y、pred_z、pred_pitch、pred_yaw。
   不要在模型输入文件中放置 GT pose。
9. 训练阶段eval interval和checkpoint save interval设置为总训练steps的 1/5；
   即训练阶段只eval和save五次，且使用late和best链接到checkpoints的最后一次保存结果和最优保存结果。
10. 训练结束后根据训练日志的主loss绘制loss曲线图。
11. 在训练时的eval和推理时增加样本可视化功能，可视化功能的主要行为有：
    - 每张地图固定随机选择 10 个样本，保证各次 eval 可横向比较。
    - Radar 上：
        - GT：同色实心圆。
        - Prediction：同色大号空心圆。
        - GT 与预测之间绘制连线。
        - 越界预测贴边显示。
    - 右侧 10 张 FPV 竖排，左上角显示对应颜色圆点。
    - FPV 顶部居中显示：
        - gt_xyzhw
        - pred_xyzhw
    - 数值使用物理坐标，xyzhw = x,y,z,pitch,yaw，角度单位为度。

如果 MODEL_TYPE=GENERATION：

1. 输入为对应 radar/map 图像和数值5DoF pose，输出对应的448×448 RGB第一视角图像。
2. 优先使用项目已有的 image condition、control image、multimodal或camera condition接口。
3. 如果原模型只有文本条件，以最小方式增加 radar encoder/adapter 和数值 pose 投影；pose 至少通过 numeric tokens、MLP、FiLM或等价数值条件接入，不能只把坐标拼成自然语言。
4. 使用 seen_train 训练、seen_validation 选 checkpoint。
5. 同一个冻结 checkpoint 分别推理：
   - seen_discrete_test
   - seen_continuous
6. continuous 按 manifest 中的 clip_id 和 frame 顺序逐帧生成，不允许读取目标帧、历史/未来GT帧。
7. 输出文件名必须保持 manifest 中的 sample/frame identity：
   outputs/csgo_benchmark_v2_seen10/${MODEL_NAME}/seed_<seed>/discrete/gen_imgs/<map>/<frame>.jpg
   outputs/csgo_benchmark_v2_seen10/${MODEL_NAME}/seed_<seed>/continuous/gen_imgs/<map>/<frame>.jpg
8. 保存图像时复用 UniLIP 当前输出尺寸、RGB转换和编码方式。
9. 训练结束后根据训练日志的主loss绘制loss曲线图。
10. 训练阶段eval interval和checkpoint save interval设置为总训练steps的 1/5；
   即训练阶段只eval和save五次，且使用late和best链接到checkpoints的最后一次保存结果和最优保存结果。

不要求不同模型使用完全相同的 optimizer、native resolution、LoRA策略或可训练参数量。优先选择当前项目最稳定、改动最小的官方训练路径，但数据 split、任务输入输出和最终 metric 必须一致。

四、需要落地的最小文件

在当前项目内新增尽可能少的内容，例如：

- 一个 manifest-driven Seen-10 dataset adapter。
- 一个模型任务适配模块。
- 一份 localization 或 generation 配置。
- train_seen10.py 或原训练入口的轻量 wrapper。
- infer_seen10.py 或原推理入口的轻量 wrapper。
- scripts/run_csgo_seen10.sh，支持 train、infer、eval 和 --seed。
- 简短的 CSGO_SEEN10.md，只记录环境、实际命令和输出路径。

不要复制完整 UniLIP 训练代码，不要重写当前项目已有的训练框架。

五、首个项目建立一次通用评测器

当 BUILD_SHARED_EVALUATOR=1 时，在 SHARED_EVAL_DIR 中实现一个与模型项目无关的评测目录。后续项目只需手动同步整个目录和配置，不再重复开发。

通用评测器应：

1. 只接收标准化的 localization JSONL 或 generation gen_imgs 目录，不 import 任何外部模型代码。
2. 从 DATA_ROOT 的 manifest、splits 和 calibration join GT。
3. 参考并复用 UniLIP 当前 metric 的算法和默认参数，只保留 Table 1 必需指标：
   - Localization：XY_Dist、Z_Dist、Pitch_Dist、Yaw_Dist。
   - Discrete：PSNR、SSIM、LPIPS、Boundary_F1、FID。
   - Continuous：PSNR、SSIM、LPIPS、TWE、TDE、FVD。
4. yaw 使用正确的最短圆周距离：
   abs(((pred - gt + period/2) % period) - period/2)
   不要照搬可能在预测越界时产生负误差的旧 min(d, period-d) 写法。
5. 连续评测固定：
   clip_length=16、clip_stride=16、fvd_size=224、
   frame_diff_threshold=2、min_track_len=4。
6. 每地图分别计算，再按上述固定地图顺序输出 equal-map macro。
7. 可以移除 Table 1 不使用的 external locator、IS、CLIP和Aesthetic，减少依赖，但不能改变保留指标的计算实现。
8. 至少提供以下统一入口：
   python run_eval.py localization --pred-root ... --data-root ... --output ...
   python run_eval.py discrete --pred-root ... --data-root ... --output ...
   python run_eval.py continuous --pred-root ... --data-root ... --output ...
9. 默认使用 UNILIP_PYTHON 运行 metric，并提供唯一一份 requirements/config。
10. 输出 per-map JSON 和 summary_equal_map.json。

当 BUILD_SHARED_EVALUATOR=0 时：

- 不重新实现评测器。
- 尽量减少修改通用评测器，如必需修改则说明并记录修改理由和内容。
- 使用已手动同步到 SHARED_EVAL_DIR 的通用代码。
- 只保证当前模型的预测满足上述标准输出格式，然后直接运行统一评测命令。
- 如果为了当前模型的预测满足上述标准输出格式而修改了 SHARED_EVAL_DIR 的通用代码，
  需要确保修改后的代码兼容之前所有 MODEL_TYPE=VLA 和 MODEL_TYPE=GENERATION 的模型预测输出格式。

六、执行要求

1. 使用当前模型自己的独立环境训练和推理；不要把它的依赖安装进 UniLIP conda 环境。
2. 先完成最小 smoke：
   - dataset能读取一个batch；
   - 模型能执行一次forward/backward；
   - 能加载保存的checkpoint；
   - 能生成一个标准预测文件或一张448×448图片；
   - 通用评测器能读取该输出。
3. smoke 失败时直接定位并修复，不能只报告问题。
4. RUN_FULL=1 时，在 smoke 后直接按 TRAIN_SEEDS 依次执行完整训练、验证、推理和评测。
5. 默认先跑 seed=0 得到完整结果；但脚本必须支持以后不改代码直接追加 seed=1、2。
6. 不进行额外审批、许可证调研、大规模数据审计、无关重构、参数量统计、资源公平性分析或长篇实验总结。
7. 不使用 incomplete/debug coverage 生成正式结果；缺文件时应补推理对应样本。
8. 不覆盖已有输出。每个 seed 使用独立目录，训练应支持从项目原生 checkpoint 恢复。

最终只需返回：

- 实际修改/新增的文件。
- 环境创建与运行命令。
- train/infer/eval 的直接命令。
- checkpoint 和结果路径。
- 若 RUN_FULL=1，给出三项任务中该模型负责部分的 per-map 与 equal-map 结果。
- 尚未完成的唯一真实阻塞项。

不要只给建议或伪代码；需要把当前项目改到能够实际运行。






# ==================== 只修改这个配置区 ====================

TARGET_MODEL_NAME="OmniGen2"
TARGET_PROJECT_ROOT="/home/jiahao/task/OmniGen2"

# AUTO 表示由你读取项目代码、配置、日志和启动脚本后自动确定。
# Show-o、Puffin 等包含嵌套工程目录时，也应自动找到真实工作目录。
TARGET_WORKDIR="AUTO"
LEGACY_CSGO_CONFIG="AUTO"
OFFICIAL_BASE_CHECKPOINT="AUTO"

ALIGNED_EXPERIMENT_NAME="csgo_seen10_exp32gen_aligned"
TRAIN_WORLD_SIZE="AUTO"
MICRO_BATCH_PER_DEVICE="AUTO"

TRAIN_SEEDS="42"
INFERENCE_SEED=42

# 0：只完成方案、实现和 smoke，不启动正式训练/全量推理。
# 1：只有在我审核方案并明确批准执行后，才允许启动正式实验。
RUN_FORMAL=0

# ==================== 配置区结束，以下内容保持不变 ====================


你需要将 TARGET_MODEL_NAME 当前已有的 CSGO Benchmark v2 生成任务接入，继续改造成能够与 UniLIP exp32/exp32_gen 进行相对公平比较的生成基线。

这次只讨论 radar/map + pose → FPV 的生成任务。不要给目标模型增加定位任务、aux_loc_loss、perception_loss 或其他 UniLIP 论文创新模块。主对照是 generation-only 的 exp32_gen；joint generation+localization 的 exp32 是次要对照，用于展示多任务统一训练的效果。

固定参考路径：

- UniLIP 根目录：
  /home/jiahao/task/UniLIP
- exp32 配置：
  /home/jiahao/task/UniLIP/csgo_configs/exp32.yaml
- exp32_gen 配置：
  /home/jiahao/task/UniLIP/csgo_configs/exp32_gen.yaml
- UniLIP 实际启动记录：
  /home/jiahao/task/UniLIP/record.md
- Benchmark v2 数据：
  /home/jiahao/task/UniLIP/data/csgo_benchmark_v2
- 共享评测器：
  /home/jiahao/task/csgo_benchmark_v2_eval_general

项目内已有的 csgo_benchmark_v2_start.md、README、旧实验说明等只能作为历史信息和现状证据，不能覆盖本 prompt 中的公平比较要求。判断实际实验设置时，证据优先级依次是：

1. 实际训练日志、trainer state、checkpoint metadata 和参数统计；
2. 实际执行的代码路径；
3. 配置文件；
4. README、接入文档和注释。

## 第一阶段：只读审计并提交修改方案

第一阶段不要修改代码、不要下载大权重、不要启动或停止训练/推理进程。先完成以下工作并等待我审核：

1. 阅读目标仓库的 AGENTS.md、git 状态、现有 CSGO 配置、训练/恢复、推理、评测脚本、日志、checkpoint 和输出目录。
2. 检查当前是否有训练、推理或评测进程，以及输出完成度。任何已有任务和结果都不能被停止、覆盖、续写或混入新实验。
3. 从 UniLIP 的配置、实际代码和日志重新核实下面给出的参考事实，不要只复述本 prompt。
4. 分析目标模型中各模块的真实功能，建立 UniLIP 模块到目标架构的“职能映射”，不能只按照类名或参数名机械匹配。
5. 给出文件级修改方案、兼容方案、训练预算计算、命令和验收标准，然后停止，等待我明确批准。

## UniLIP 参考实验

需要重新核实并以实际证据为准的参考设置如下。

### exp32_gen

- generation-only，Seen-10 `seen_train` 共 50,000 条。
- 单卡 batch 128，梯度累计 1，有效 generation batch 128。
- 50 epochs，实际完成 19,500 个 optimizer updates。
- generation 样本曝光量为：

  19,500 × 128 = 2,496,000

- 冻结视觉塔、VAE 和 inactive localization 分支。
- 活跃 LLM 使用 LoRA：
  - rank 32
  - alpha 64
  - dropout 0.05
  - attention 的 q/k/v/o 投影
  - MLP 的 gate/up/down 投影
- generation DiT/generation expert 使用 LoRA：
  - rank 32
  - alpha 64
  - dropout 0.05
  - attention q/k/v/out 和 MLP 对应线性层
- generation connector 使用 LoRA：
  - rank 16
  - alpha 64
  - dropout 0.05
- latent queries、generation projector 等小型生成模块全量训练。
- 主学习率 1e-4，AdamW，weight decay 0，warmup ratio 0.003，`cosine_with_min_lr`，最低学习率 1e-5。
- 原始训练 `eval_strategy=no`，正式结果使用训练结束时的 final model，而不是 validation-best。
- 条件包括 radar/map、地图名称和 5DoF pose；生成目标为 448×448 FPV。
- 条件视觉输入实际按 224×224 处理；生成目标保持 448×448。
- 没有随机 crop、颜色扰动、随机擦除等图像数据增强。
- 使用模型原生 flow-matching 训练目标。

### exp32

- joint generation+localization。
- 每次 optimizer update 使用 128 条源记录，并展开成约 128 个 generation 样本和 128 个 localization 样本。
- 实际约 19,550 updates，因此 generation 样本预算与 exp32_gen 基本一致。
- 它是联合任务结果，不能用来替代 generation-only 的 exp32_gen 作为外部生成模型的首要公平对照。

## 必须对齐的实验条件

### 1. 数据、输入信息和目标

使用同一份 Benchmark v2 manifest、selection、split 和 calibration，不允许重新扫描目录自行构造 split。

固定数据规模：

- `seen_train`：50,000
- `seen_validation`：5,000
- `seen_discrete_test`：20,000
- `seen_continuous`：12,800
- 连续集为 200 clips × 64 frames

固定 Seen-10 地图：

- cs_agency
- cs_italy
- de_ancient
- de_anubis
- de_dust2
- de_inferno
- de_mirage
- de_nuke
- de_overpass
- de_train

模型只能使用：

- 当前样本的 radar/map 图；
- map identity；
- 当前样本的 x、y、z、pitch、yaw；
- 固定的任务文本或模型原生条件表示。

不允许使用：

- 测试样本的目标 FPV；
- 邻近、历史或未来真实 FPV；
- 连续序列中的前一真实帧；
- 由前一生成帧形成的额外时序条件；
- 测试集指标进行 checkpoint 或超参数选择。

如果目标模型使用数值 pose adapter，归一化必须来自发布协议：

- x / 1024
- y / 1024
- z 使用发布的逐地图 frozen exact min/max
- pitch / (2π)
- yaw / (2π)

如果模型原生使用文本坐标，可以使用对应物理数值文本。无论使用文本、数值 adapter 或两者组合，包含的信息必须相同，并在报告中明确说明条件注入方法和额外参数量。

radar 条件输入默认对齐到有效 224×224，目标 FPV 为 448×448。若模型固定结构确实无法使用 224×224 条件输入，需要给出代码证据和最接近的兼容方案，作为待审核例外，不能静默保留 448 或其他分辨率。

训练和验证可以读取 target FPV；离散和连续推理的数据路径必须验证为 `load_target=False` 或等价行为。

### 2. 训练预算

新实验固定为：

- 有效 generation batch：128
- optimizer updates：19,500
- generation 样本曝光量：2,496,000

必须满足：

effective_generation_batch
= world_size × micro_batch_per_device × gradient_accumulation
= 128

显存不足时使用梯度累计。CFG 条件/无条件分支、同一样本的多个内部表示或模型内部 token 展开不能重复计作独立 generation 样本。

以 `max_optimizer_steps=19500` 作为权威终止条件，不允许因为项目原来的 epoch、iteration 或 sample counter 语义不同而改变预算。方案中必须列出：

- world size；
- 单卡 micro batch；
- gradient accumulation；
- 每个 optimizer step 的真实源样本数；
- 总 optimizer steps；
- 总样本曝光量；
- 等价 epoch 数；
- scheduler 是否按 optimizer step 更新。

如果现有训练循环不正确缩放 accumulation loss、在 micro step 更新 scheduler、或把 micro step 记作 global step，需要修正。

### 3. 可训练模块与 LoRA

目标不是让不同架构拥有完全相同的参数数量，而是对齐模块职能和参数高效微调强度。

默认映射规则：

1. 冻结预训练视觉/radar encoder。
2. 冻结 VAE、VQ tokenizer 或其他图像 tokenizer。
3. 对直接参与生成条件理解的 LLM/语言 backbone：
   - LoRA r32、alpha64、dropout0.05；
   - 覆盖 attention q/k/v/o 和 MLP gate/up/down 的等价层。
4. 对主要生成 Transformer、DiT、GPT 或 generation expert：
   - LoRA r32、alpha64、dropout0.05；
   - 覆盖 attention 和 MLP 的等价线性层。
5. 对预训练 connector/bridge：
   - LoRA r16、alpha64、dropout0.05。
6. 以下新建或小型生成模块可以全量训练：
   - radar/pose/map/task adapter；
   - generation projector/output bridge；
   - latent queries、meta queries 或等价查询参数。
7. 冻结理解、定位及其他未参与本生成任务的分支。

如果目标模型只有一个统一生成 Transformer，不要把它重复映射为 LLM 和 DiT 并注入两套 LoRA。应根据真实计算图映射一次。

若模型使用 fused QKV、MoE、卷积投影或其他结构，使用数学上最接近的 LoRA/PEFT 注入点。不得未经说明继续使用：

- 大规模 generation backbone 全量训练；
- 只训练一个很小的新 adapter 而完全冻结生成 backbone；
- 与参考明显不同的 r8、dropout0 等设置。

如果某项无法实现，应给出具体技术原因、涉及参数名和最接近的替代方案，等待审核。

方案必须输出一张模块表，至少包含：

- UniLIP 参考职能；
- 目标模型模块和精确参数名前缀；
- frozen/full/LoRA 状态；
- LoRA 参数；
- 学习率组；
- trainable 参数量；
- 占总参数比例；
- 选择依据。

实现后必须打印并保存完整 trainable-parameter audit，证明冻结模块没有被 optimizer 收录。

### 4. 优化与数据处理

公平配置默认采用 exp32_gen 优化设置：

- AdamW
- β=(0.9, 0.999)
- weight decay=0
- learning rate=1e-4
- warmup ratio=0.003
- cosine decay
- minimum learning rate=1e-5
- scheduler 按 optimizer step 更新

所有 LoRA 和新建小型条件模块默认使用 1e-4。若目标架构存在有代码证据的数值不兼容，应先在方案中提出，不要直接改变正式主实验。可以额外提出 model-native LR sensitivity，但不能代替对齐主配置。

保留模型原生生成目标：

- AR 模型继续使用 token cross-entropy；
- diffusion/flow 模型继续使用原生 noise、velocity 或 flow-matching 目标；
- 保留官方 VAE/VQ、噪声参数化和 timestep 定义。

不要为了统一 loss 类型而重写模型。

图像只做模型必需的确定性 resize、归一化和 tokenizer/processor 处理。关闭随机 crop、flip、颜色增强、grid/coarse dropout、random erasing 等随机图像增强。

对于需要 CFG 训练的模型，条件 dropout 默认对齐为 0.1；不使用 CFG 的模型不要人为增加无条件分支。必须记录实际行为。

### 5. checkpoint 与选择规则

整个正式训练只设置五个常规保存和完整验证里程碑：

- step 3,900
- step 7,800
- step 11,700
- step 15,600
- step 19,500

每次在完整 `seen_validation` 5,000 条上计算目标模型的原生 validation generation loss。

产出两个稳定别名：

- `best`：五次 validation loss 最小的 checkpoint；
- `late`：step 19,500，即训练结束 checkpoint。

不同模型的 loss 数值不能跨模型直接比较，只用于同一模型内部选点。

严格论文主比较使用 `late/final`，因为现有 UniLIP exp32/exp32_gen 正式结果来自 final model。`best` 作为补充结果报告。如果希望把 best 作为主结果，必须先对 UniLIP 应用同样的 validation 选点规则，不能用外部模型 best 对比 UniLIP final 并称为严格同规则比较。

checkpoint 必须支持精确恢复：

- 模型及 LoRA；
- 全量训练的小模块；
- optimizer；
- scheduler；
- AMP scaler；
- global optimizer step；
- RNG；
- sampler/dataloader epoch 状态；
- 梯度累计边界。

### 6. 推理与评测

`seen_discrete_test` 和 `seen_continuous` 必须使用同一个冻结 checkpoint。

要求：

- 每个 condition 只生成一张图；
- 不允许 best-of-N 或按 GT 选择样本；
- 固定 inference seed；
- 随机数最好由 seed + sample_id 派生，使 batch size 和断点续跑不改变单样本结果；
- 连续集逐帧独立生成，但保留原始 clip/frame identity 和顺序；
- 输出严格为 448×448 RGB；
- 文件名、目录、manifest identity 和图像编码参数对齐共享协议及 UniLIP 实际 saver；
- 推理恢复时校验已有文件完整性，不能把其他 checkpoint 或旧实验的图片混入。

保留各模型预先声明的原生采样器、scheduler 和 guidance 机制，因为 AR、flow 和 diffusion 的采样过程不可机械统一。所有推理参数必须在查看测试指标前确定，并报告：

- sampler；
- steps/NFE；
- guidance/CFG；
- time shift；
- VAE/VQ；
- precision；
- inference seed。

对于能够合理运行 20 NFE 的 diffusion/flow 模型，可以提出额外的 compute-matched 20-NFE 配置，但不能未经审核替换模型原生主配置。

只能使用共享评测器：

/home/jiahao/task/csgo_benchmark_v2_eval_general

离散评测包括：

- PSNR
- SSIM
- LPIPS
- Boundary_F1
- FID

连续评测包括：

- PSNR
- SSIM
- LPIPS
- TWE
- TDE
- FVD

使用共享配置规定的 equal-map macro、clip length 16、stride 16、FVD size 224、frame-difference threshold 和 tracking 参数。不要在目标项目中复制或修改指标实现。

### 7. 兼容性和结果隔离

新增独立的 aligned config/profile，例如：

csgo_seen10_exp32gen_aligned

不能直接修改旧配置的含义。现有训练、推理、评测命令仍应能够复现旧实验；通过新增 `--experiment`、`--config` 或等价参数显式选择 aligned 实验。

新实验必须：

- 从官方原始基础权重开始；
- 不从当前已经训练过的 CSGO checkpoint 初始化；
- 使用独立 run root、checkpoint、prediction、manifest、evaluation 和日志目录；
- 不覆盖 best、late、latest 等旧链接；
- 不续写已有不完整推理目录；
- 不停止当前运行的训练或推理任务。

## 第一阶段回复格式

第一阶段只给修改方案，至少包含：

1. 当前项目实际状态和仍在运行的进程；
2. `exp32`、`exp32_gen`、当前旧配置、拟议 aligned 配置的对照表；
3. 有效 batch、累计步数、更新数和曝光量的完整计算；
4. UniLIP 到目标架构的模块职能映射表；
5. 精确 LoRA target module 名称、参数量及比例；
6. 优化器和每个参数组的 LR；
7. 数据、条件输入、分辨率、增强和 target 隔离检查；
8. checkpoint 主/次报告规则；
9. 需要修改、新增的文件及每个文件的作用；
10. 保持旧命令兼容的方法；
11. 新训练、resume、best/late 离散推理、连续推理和评测命令；
12. 预计训练时间、磁盘占用和主要风险；
13. 实现后的 smoke 和验收步骤。

引用结论时提供绝对路径和行号。完成方案后停止，不修改文件，等待我审核。

## 审核批准后的执行边界

只有在我明确批准方案后才实施代码变更。实施后完成：

- 配置解析检查；
- split、地图、样本数和 calibration 检查；
- 推理 target FPV 隔离检查；
- trainable/frozen 参数审计；
- LoRA target 覆盖检查；
- 梯度累计和 optimizer-step 语义检查；
- 独立 smoke 输出目录中的少量训练与精确 resume 测试；
- 离散、连续各少量样本推理；
- 共享 evaluator 的 smoke 检查；
- 旧命令兼容检查。

`RUN_FORMAL=0` 时，完成实现和 smoke 后停止，不启动 19,500-step 正式训练，也不启动全量 20,000/12,800 样本推理。






你现在负责为一个 VLA 模型设计 CSGO Benchmark v2 定位任务的公平对比实验。

目标项目：<目标项目路径或名称>
目标分支：<目标分支>
目标模型：<模型名称>
原始预训练权重：<checkpoint>
UniLIP 项目路径：<UniLIP 路径；若未填写，请在当前目录及相邻目录中查找>
CSGO Benchmark v2 manifest：<benchmark_manifest.json 路径>

本轮只进行代码、配置、日志和 checkpoint 元数据审计，并给出具体变更方案。不要修改代码、下载大模型或启动正式训练。方案完成后等待我审核。

## 一、任务目标

需要让目标 VLA 模型接入 CSGO Benchmark v2 的定位训练、推理和评测，并与 UniLIP 的以下实验进行相对公平的比较：

1. `exp32_loc`：定位-only 对照，是其他 VLA 定位基线的主要比较对象。
2. `exp32`：生成+定位联合训练模型，其定位结果作为次要比较对象，用于判断 multi-task 是否有益。
3. 生成任务、`exp32_gen` 和其他生成模型不在本次工作范围内。

这里的“公平”不是机械复制 UniLIP 的每一个内部实现，而是：

- 数据、输入信息、目标定义、训练样本暴露量、评测协议和 checkpoint 选择必须尽量一致；
- 模型内部结构和必须保留的原生训练机制可以不同；
- 所有不能对齐的地方必须明确披露，并分析其可能造成的优势或劣势；
- 不允许向目标模型提供 UniLIP 没有使用的额外状态、标签、地图坐标或测试集统计。

## 二、必须先核实的 UniLIP 参考事实

不要只相信实验名、注释或 YAML。请同时检查配置、训练入口、数据集、模型前向、优化器参数组、训练日志、保存的 `config.json`、`training_args.bin`、`trainer_state.json` 和 scheduler state，区分：

- 配置文件声明值；
- CLI 传入值；
- 合并后的运行时值；
- optimizer 实际参数组及学习率；
- checkpoint 中保存的最终值。

当前已经掌握的参考事实如下，但仍需用本机 UniLIP 代码和产物复核；若仓库实际状态不同，必须报告差异。

### 1. 数据和任务

- `exp32` 和 `exp32_loc` 使用 Benchmark v2 的 Seen-10 `seen_train`。
- 输入是 FPV、对应地图 radar 和文本指令；文本中允许包含地图名称。
- 定位输出定义为单步 5D pose：`x, y, z, pitch, yaw`，`action_horizon=1`。
- `exp32` 是 balanced joint 训练，每条源样本产生一个定位样本和一个生成样本。
- `exp32_loc` 是 localization-only，`task_mix_ratio=1.0`。
- VLA 定位基线不需要承担生成任务；主要应与 `exp32_loc` 比较，同时报告与 `exp32` 定位结果的差异。

### 2. Pose normalization

UniLIP 的 exp32/exp32_loc 不使用 OpenPI 风格的 q01/q99 quantile normalization。其外部 5D 标签为：

x'     = x / 1024
y'     = y / 1024
z'     = (z - z_min_map) / (z_max_map - z_min_map + 1e-6)
pitch' = pitch / (2π)
yaw'   = yaw / (2π)

其中：

- `z_min_map/z_max_map` 来自 Benchmark v2 发布的、按地图冻结的 exact min/max；
- 该范围在数据划分前根据 approved full corpus 计算；
- train/test/cross-map 使用同一套发布标定；
- 不从测试集重新计算，不做隐式 clamp；
- benchmark 构造中的 quantile bins 只用于样本覆盖抽样，不是 action quantile normalization。

如果目标 VLA 原生依赖 quantile normalization：

- 优先设计使用上述官方 5D normalization 的严格对齐实验；
- 如果直接取消原生 normalization 会破坏预训练 action head，需要同时提出“严格对齐”和“原生 normalization”两个可区分方案；
- 原生 quantile stats 只能由 `seen_train` 计算，禁止使用 validation/test；
- 必须说明哪一个作为论文主结果，哪一个作为消融或模型原生对照。

### 3. UniLIP 的 π0.5 localization head

- 对外 `action_dim=5`。
- π0.5 action head 内部使用 32D。
- 训练时把 5D pose 后补 27 个零，形成 32D flow-matching target。
- 噪声、velocity prediction 和主 MSE 都在完整 32D 上计算。
- `loc_loss_valid5` 仅是监控指标，不替代 32D 主损失。
- 推理从 32D 噪声开始，最后只返回前 5 维。
- 对于 π0.5 或同类 action expert，应保留这一 32D 路径。
- 对于内部结构不同的 VLA，不应为了表面一致强制改成 32D；应保持模型原生 head，在外部统一为相同的 5D pose、normalization 和评测协议。

### 4. State token

UniLIP 的这条定位路径没有独立 state 输入，也没有实际接入 state token。它不是 YAML 中一个简单的 `state_token=false` 开关，而是模型路径中根本没有把 state token 拼入前缀或 action suffix。

目标实验必须：

- 默认关闭或彻底省略 state token；
- 不要用“全零 state token”冒充关闭，因为额外 token 仍可能改变注意力和序列结构；
- 如果目标架构强制要求 state token，应使用 mask/零输入的最小影响方案，并明确报告；
- 不得输入 GT 坐标、历史位姿或其他 UniLIP 没有使用的状态信息。

### 5. 图像处理和增强

UniLIP exp32/exp32_loc 的实际随机图像增强关闭：

- `is_fps_dropout` 没有启用；
- CoarseDropout、GridDropout、RandomErasing 虽然在代码中存在，但本实验没有执行；
- 测试阶段也没有随机增强。

实际定位输入大致为：

- RGB；
- InternVL processor 确定性 resize/rescale/normalize；
- 输入定位 backbone 的图像尺寸为 224；
- 不使用随机 crop、rotation、ColorJitter、CoarseDropout、GridDropout 或 RandomErasing。

目标 VLA 若支持 224 输入，严格对齐实验应使用 224。若模型预训练结构固定为其他分辨率，保留原生分辨率并报告，同时判断是否需要补充 224 分辨率消融。

### 6. 可训练模块和 LoRA

`exp32_loc` 的定位路径：

- vision tower：冻结；
- InternVL multimodal projector：冻结；
- LLM：base 冻结，LoRA 可训练；
- action expert：base 冻结，LoRA 可训练；
- localization action connector：全量可训练；
- action norm/projector：全量可训练；
- action input/output projection和 timestep MLP：全量可训练；
- generation connector、generation DiT、generation projector、latent queries：冻结。

LoRA 参考设置：

- `r=32`
- `alpha=64`
- `dropout=0.05`
- `bias=none`
- LLM/action expert 目标层：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

`fix_connect=True` 和 `fix_dit=True` 在 exp32_loc 中主要指 generation 分支，不能据此误判 localization action connector 或 action expert 也被冻结。必须通过最终 `requires_grad` 清单和 optimizer 参数组确认。

对其他 VLA 应按“功能角色”映射模块，而不是按名字硬套：

- vision encoder；
- language backbone；
- vision-language connector；
- action expert/policy backbone；
- action input/output head；
- timestep/noise embedding；
- action connector、norm 或 projector。

如果目标模型没有某个模块，明确标为“不适用”。同时报告总参数量、可训练参数量、比例及各组 LR。必要时提出：

1. 功能角色对齐方案；
2. 可训练参数量近似对齐方案。

指定其中一个作为主实验，避免事后选择更好的结果。

### 7. 学习率参考

UniLIP 的实际参考 LR 为：

| 模块 | exp32 | exp32_loc |
|---|---:|---:|
| LLM LoRA | 1e-4 | 1e-4 |
| Action Expert LoRA | 5e-5 | 1e-4 |
| Action connector | 1e-4 | 5e-4 |
| Action norm | 5e-4 | 5e-4 |
| Action projector | 5e-4 | 5e-4 |
| Action input/output + timestep MLP | 1e-4 | 1e-4 |

公共设置：

- AdamW
- weight decay 0
- warmup ratio 0.003
- cosine with minimum-LR schedule
- 全局 `min_lr=1e-5`

不要默认认为所有不同架构都必须使用同一个 LR。请判断：

- 哪些模块和 UniLIP 结构对应，可以直接继承 LR；
- 哪些模块应使用目标模型已有的稳定训练 LR；
- 是否需要一个预先固定的小规模 LR sweep；
- sweep 只能依据 validation 结果选择，并保证不同 baseline 的搜索预算一致。

### 8. Batch 和训练预算

参考训练预算：

- exp32：2 GPU × 每卡 batch 4 × 梯度累计 16，即每次 update 为 128 条源数据；balanced 展开后是 128 个 loc + 128 个 gen task samples；实际约 19,550 optimizer updates。
- exp32_loc：1 GPU × batch 128 × 累计 1，即每次 update 为 128 个定位样本；实际 19,500 optimizer updates。
- 两者训练 50 epochs，训练集约 50,000 条源数据。

VLA localization-only 主实验优先对齐 exp32_loc：

- effective localization batch size = 128；
- optimizer updates = 19,500；
- batch 不够时使用梯度累计；
- 对齐的是 optimizer update 和定位样本暴露量，不是单卡 microbatch；
- scheduler、warmup 和 checkpoint 间隔按 optimizer update 计算；
- 禁止因为更换 GPU 数量而改变有效 batch 或总更新数。

### 9. 初始化和 checkpoint 选择

- 目标模型必须从公开/原始预训练权重开始。
- 不得从已经见过 CSGO Benchmark v2 train/test 的 checkpoint warm-start。
- π0.5 使用原始 `pi05_base`；其他 VLA 使用各自官方 base checkpoint。
- 必须记录 checkpoint 来源和哈希或可验证标识。

主比较应使用预先规定的 checkpoint 规则：

- `late`：最后一个或训练结束 checkpoint；
- 若报告 `best`，只能依据 validation loss/metric 选择，不能依据 test；
- 为了直接对应 UniLIP final checkpoint，论文主表至少报告 `late/final`；
- 如果所有模型统一报告 best-val，可以作为附加结果；
- 建议全程等间隔保存约 5 次，避免因模型不同而获得不同密度的 checkpoint 搜索机会。

### 10. 推理和评测

必须复用 Benchmark v2 官方 manifest 和评测脚本：

- Seen-10：`seen_discrete_test`；
- 如论文需要，补充 CrossMap-4 `crossmap_query_test` zero-shot；
- 使用相同地图、样本 ID、pose 反归一化和 metric 聚合方式；
- 同时报告 pooled 指标和 equal-map macro 指标；
- 至少报告 XY、Z、Pitch、Yaw；
- 不得通过测试集选择 checkpoint、LR、normalization stats 或推理超参数；
- 随机种子、推理步数、采样方式和约束必须记录。
- 不同模型的解码算法可以保留原生实现，但要报告推理步数和额外计算成本。

## 三、你需要执行的审计

请读取目标项目与 UniLIP，至少完成：

1. 找到目标模型当前的数据加载、action normalization、state 构造、图像预处理、loss、推理反归一化和评测入口。
2. 追踪训练配置从 CLI/YAML 到运行时对象的覆盖顺序。
3. 输出最终可训练参数清单或按模块统计，并检查 optimizer 实际 LR。
4. 检查当前模型是否：
   - 使用了 quantile norm；
   - 输入了 state token；
   - 内部 action 维度不是 5；
   - 使用额外数据增强；
   - 使用不同训练样本、地图或 split；
   - 从旧 CSGO checkpoint 恢复；
   - 训练步数或有效 batch 不一致；
   - 在测试集上选择 checkpoint；
   - 使用额外监督或额外输入。
5. 读取已有训练日志和 checkpoint metadata，不能只依据源代码推测。
6. 对任何缺失证据标记“未确认”，不要把注释、默认值或相邻实验当作实际运行值。

## 四、公平性分层

请把每个配置项归入以下一类：

### A. 必须严格对齐

通常包括：

- manifest、split、地图和样本；
- 输入信息边界；
- 外部 5D pose 定义和物理单位；
- train/test normalization contract；
- state token 关闭；
- 随机增强策略；
- 定位有效 batch；
- optimizer update 数和定位样本暴露量；
- checkpoint 选择；
- 官方评测脚本和指标。

### B. 按模型原生结构适配

通常包括：

- 内部 action latent 维度；
- regression、autoregressive 或 flow-matching objective；
- tokenizer；
- 模型固定输入分辨率；
- 原生 action expert 结构；
- 必须保留的优化器稳定性设置。

这些差异必须有理由，并分析是否会影响比较。

### C. 只需报告和控制

通常包括：

- 总参数量和可训练参数量；
- 预训练语料；
- FLOPs、显存、训练时间；
- 推理步数和延迟；
- 模型原生结构造成的不可消除差异。

## 五、输出格式

请按以下结构给出结果：

1. **结论**
   - 当前目标实验是否已经能和 exp32_loc 公平比较；
   - 最大的 3～5 个不一致项；
   - 推荐的主实验定义。

2. **证据化对齐矩阵**

   表格必须包含：

   | 对齐项 | exp32 | exp32_loc | 目标项目当前值 | 建议值 | 分类 A/B/C | 证据文件/行号 |

3. **目标模型模块映射**
   - vision、LLM、connector、action expert、action head、timestep module 的对应关系；
   - 每个模块冻结、LoRA 或全量训练；
   - LoRA target、rank、alpha、dropout；
   - 各模块 LR；
   - 总参数量和可训练参数量。

4. **Normalization 和数据流**
   - 从原始 JSON pose 到训练 target；
   - 模型内部表示；
   - loss 计算维度；
   - 推理输出；
   - 反归一化和 metric；
   - 明确说明是否存在 qnorm、clamp、state token 或测试集统计泄漏。

5. **具体变更方案**
   - 精确到文件、类、函数和配置字段；
   - 说明新增配置开关及默认值；
   - 保持现有训练/推理/评测命令兼容；
   - 旧实验必须仍可通过旧配置复现；
   - 新实验必须从原始 base 权重开始；
   - 不要把目标模型改造成 UniLIP，只实现公平比较需要的最小变更。

6. **实验矩阵**
   至少包含：
   - 现有实现复现；
   - 严格公平的 localization-only 主实验；
   - 只有在必要时才增加 normalization、state 或内部 action 维度消融。

   为每项列出：
   - 初始化；
   - split；
   - trainable modules；
   - normalization；
   - augmentation；
   - effective batch；
   - updates；
   - checkpoint 选择；
   - 推理和评测命令。

7. **验收标准**
   - 数据 ID 和数量检查；
   - target 数值和反归一化 round-trip 检查；
   - state token 确实不存在或被 mask；
   - trainable parameter 与 optimizer LR 审计；
   - 单 batch 前向/反向；
   - 小规模 overfit 或 smoke test；
   - 推理输出和官方 evaluator 闭环；
   - 不启动正式 19,500-step 训练。

8. **风险和仍未确认项**
   - 只列有证据的风险；
   - 明确哪些差异是模型原生、无法完全消除；
   - 给出对论文表述的影响。

本轮完成上述分析和变更方案后停止，等待我审核。未经审核不要修改代码，也不要启动正式训练。