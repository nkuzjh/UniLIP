我检索后，比较值得参考的不是一类，而是几条路线。严格说，最像你现在“定位理解任务辅助生成”的，是 **MT-Diffusion / ControlNet++ / OneDiffusion** 这几类；其他如 ControlNet、DALL-E 3 recaptioning、Janus/Show-o 更偏条件控制、数据增强或统一架构。

| 方向 | 代表工作 | 核心思路 | 对当前 CSGO 的启发 |
|---|---|---|---|
| 多任务 diffusion | [MT-Diffusion: Diffusion Models for Multi-Modal Generative Modeling](https://assets.amazon.science/a9/b7/bf71d8ae4feaa4c7d744877057b4/diffusion-models-for-multi-modal-generative-modeling.pdf) | 在统一 diffusion space 中联合建模图像、标签、mask、CLIP 表征等多模态/多任务目标；论文里还做了 masked-image training 来提高生成训练效率。 | 很接近“gen + loc”联合训练。可以参考它的 shared backbone + task-specific head / decoder，以及多任务 loss 如何进入 diffusion 训练。 |
| 判别模型反馈生成 | [ControlNet++](https://arxiv.org/abs/2404.07987) | 用预训练判别/理解模型从生成图中重新提取条件，比如 segmentation/depth/edge，再和输入条件做 consistency loss；还提出单步 denoise 的高效 reward fine-tuning，避免完整采样反传太贵。 | 非常贴近你的 aux-loc 思路：把 loc head 当作 reward/consistency model，从生成 FPS 预测 pose，再和 GT pose 或 teacher pose 对齐。 |
| 统一生成+理解 diffusion | [One Diffusion to Generate Them All](https://arxiv.org/abs/2411.16318) | 一个 diffusion 模型同时做 text-to-image、depth/segmentation/camera pose estimation、multiview generation 等；把任务统一成 frame sequence，不同 frame 注入不同 noise scale。 | 很适合参考“定位/深度/pose estimation 作为理解任务”和“图像生成作为反向任务”如何放进同一个训练格式。 |
| 结构条件控制 | [ControlNet](https://arxiv.org/abs/2302.05543), [T2I-Adapter](https://arxiv.org/abs/2302.08453), [GLIGEN](https://arxiv.org/abs/2301.07093) | 用 pose、depth、segmentation、edge、bbox 等理解任务输出作为生成条件。ControlNet/T2I-Adapter 加可训练控制分支；GLIGEN 注入 grounding/bbox 条件。 | 如果不想只用 loc loss，可以把 CSGO pose / map-relative loc 显式编码成条件 token 或 spatial condition，直接控制 FPV 生成。 |
| 控制一致性奖励 | ControlNet++ 同上 | 比 ControlNet 更进一步，不只输入 condition，还检查生成图是否真的满足 condition。 | 这条和当前 exp27/aux-loc 最相关：可以做 `generated FPS -> loc head -> pose` 与 GT pose 的 cycle consistency，甚至只在低噪声 step 上做。 |
| 任意理解模型 guidance | [Universal Guidance for Diffusion Models](https://arxiv.org/abs/2302.07121) | 不重新训练 diffusion，用任意 guidance function 控制生成，包括 segmentation、face recognition、object detection、classifier 信号。 | 可参考“loc head gradient 作为 guidance”的形式。训练时是 aux loss，采样时也可做 localization-guided sampling。 |
| 分类器/判别器 guidance | [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233), [GLIDE](https://arxiv.org/abs/2112.10741) | 用 classifier guidance 或 classifier-free guidance 提高 sample quality 和 condition fidelity。 | 原理上说明“理解/判别信号可以显著提升生成质量”。你的 loc head 相当于更细粒度的 task classifier/regressor。 |
| VLM 生成高质量语义监督 | [DALL-E 3: Improving Image Generation with Better Captions](https://cdn.openai.com/papers/dall-e-3.pdf), [PixArt-α](https://arxiv.org/abs/2310.00426) | 用强 captioner/VLM 重新标注数据，提升文本-图像对齐和训练效率。PixArt-α 明确用 VLM dense pseudo-caption 加速训练。 | 对 CSGO 可类比为：用强定位/场景理解模型离线生成更密集的 pose、区域、可见物体、朝向、地图语义标签，作为生成训练的辅助监督。 |
| 视角/相机姿态条件生成 | [Zero-1-to-3](https://arxiv.org/abs/2303.11328), [3DiM](https://arxiv.org/abs/2210.04628) | 用相对相机 pose / target view condition 来做 novel view synthesis。 | 和 CSGO FPV 很相关：不是只让模型隐式学 pose，而是把 camera pose / player pose 明确作为 generation condition。 |
| 统一多模态理解+生成模型 | [Show-o](https://arxiv.org/abs/2408.12528), [Janus](https://arxiv.org/abs/2410.13848), [Emu3](https://arxiv.org/abs/2409.18869), [Unified-IO 2](https://arxiv.org/abs/2312.17172) | 同一模型/框架同时处理 VQA、captioning、image generation、editing、action 等任务。Janus 特别强调理解和生成视觉编码解耦，避免任务冲突。 | 可以参考架构层面：loc understanding 和 FPV generation 不一定共用全部视觉路径；可共享高层 transformer，但 decouple vision encoder/projector，减少互相拖累。 |
| 通用图像生成任务统一 | [OmniGen](https://arxiv.org/abs/2409.11340) | 把 image editing、subject-driven generation、pose recognition、edge detection、deblur 等任务统一成 image generation 格式，通过多任务数据实现知识迁移。 | 可以参考把 CSGO 中的 loc、map grounding、view reconstruction、future frame generation 都统一成“输入若干条件，输出图像/pose”的任务格式。 |

**最建议优先读的 4 个**

1. **ControlNet++**：最像你现在的 `loc_head` consistency / perceptual loss，可以直接启发 exp 系列后续设计。
2. **MT-Diffusion**：最像“生成任务 + 理解任务”联合 diffusion 训练框架。
3. **OneDiffusion**：包含 camera pose estimation、depth、segmentation、generation 的统一训练方式，和 CSGO 场景很贴。
4. **Janus / Show-o**：提醒你注意理解和生成共享模块时的任务冲突，尤其是是否要 decouple visual encoding。