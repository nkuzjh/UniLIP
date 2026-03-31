# CSGO Experiment Summary

本文件总结当前 `csgo_configs` 下近期新增的实验设计，方便在新线程继续讨论和修改。

## Current Experiments

### exp15
- 训练模式：`exp16` 风格，只训练 `gen` 样本，同时使用 `aux_loc_loss`
- 权重来源：
  - `loc head` 先通过 `--pretrain_path` 从 `exp4_12_2` 加载
  - `gen/shared` 再通过 `resume_ckpt_path` 从 `exp2` 加载
- 关键配置：
  - `is_multi_task_balanced: False`
  - `task_mix_ratio: 0.0`
  - `alpha_loc_loss: 0.0`
  - `is_loc_aux_loss: True`
  - `alpha_loc_aux_loss: 100`
  - `use_pi05_action_dit: True`
  - `is_action_dit_dense_timestep: True`
  - `is_action_dit_projector: True`
  - `action_dit_projector_lr: 0.0005`
  - `img_size: 448`
  - `is_lora: False`

### exp15_1
- 训练模式：`exp11` 风格，联合 `loc/gen` 训练，同时使用 `aux_loc_loss`
- 权重来源：
  - `loc head` 先通过 `--pretrain_path` 从 `exp4_12_2` 加载
  - `gen/shared` 再通过 `resume_ckpt_path` 从 `exp2` 加载
- 关键配置：
  - `is_multi_task_balanced: True`
  - `alpha_loc_loss: 2`
  - `is_loc_aux_loss: True`
  - `alpha_loc_aux_loss: 100`
  - 其余定位结构开关与 `exp15` 相同
  - `img_size: 448`
  - `is_lora: False`

### exp17
- 训练模式：和 `exp15_1` 相同，联合 `loc/gen` 训练，同时使用 `aux_loc_loss`
- 权重来源：
  - 仅从原始 `UniLIP-1B` 初始化
  - 不加载任何 CS:GO 训练权重
  - 不传 `--pretrain_path`
  - 不设置 `resume_ckpt_path`
- 关键配置：
  - `is_multi_task_balanced: True`
  - `alpha_loc_loss: 2`
  - `is_loc_aux_loss: True`
  - `alpha_loc_aux_loss: 0.1`
  - 其余定位结构开关与 `exp15_1` 基本一致
  - `img_size: 448`
  - `is_lora: False`

## Dynamic Alpha Variants

### exp15_2
- `exp15` 的动态 `alpha_loc_aux` 版本
- schedule:
  - `steps: [0, 500, 2000, 4500]`
  - `values: [0.0, 10.0, 50.0, 100.0]`

### exp15_3
- `exp15_1` 的动态 `alpha_loc_aux` 版本
- schedule:
  - `steps: [0, 500, 2000, 4500]`
  - `values: [0.0, 20.0, 100.0, 200.0]`

### exp17_1
- `exp17` 的动态 `alpha_loc_aux` 版本
- schedule:
  - `steps: [0, 3000, 4000, 5000, 10000]`
  - `values: [0.0, 1.0, 2.0, 5.0, 10.0]`

## Code Support

当前代码已支持按 step 动态更新 `alpha_loc_aux_loss`：

- 文件：`train_csgo.py`
- 机制：
  - 新增 `_piecewise_linear_value(...)`
  - 新增 `AlphaLocAuxScheduleCallback`
  - 训练入口会读取：
    - `alpha_loc_aux_schedule_steps`
    - `alpha_loc_aux_schedule_values`
  - 若这两个键存在，则 callback 会在训练时按 `global_step` 动态更新 `model.config.alpha_loc_aux_loss`

模型 forward 端不需要再改：
- `unified_unilip.py` 每步直接读取 `model.config.alpha_loc_aux_loss`

## Config Keys For Dynamic Alpha

如果后续继续新增动态版本，沿用下面两个键即可：

```yaml
alpha_loc_aux_schedule_steps: [0, 500, 2000, 4500]
alpha_loc_aux_schedule_values: [0.0, 10.0, 50.0, 100.0]
```

同时建议把静态初值写成：

```yaml
alpha_loc_aux_loss: 0.0
```

## Practical Notes

- `exp15 / exp15_1` 继续训练时突然加入 `aux_loc_loss`，存在生成性能波动风险
- `exp17` 从 scratch 联合训练时，前期 `aux_loc_loss` 通常更应保守，甚至先为 `0`
- 评估时不要只看 `total_loss`，应拆开观察：
  - `gen_loss`
  - `loc_loss`
  - `loc_aux_loss`
  - 当前 `alpha_loc_aux_loss`

## Useful Paths

- `csgo_configs/exp15.yaml`
- `csgo_configs/exp15_1.yaml`
- `csgo_configs/exp15_2.yaml`
- `csgo_configs/exp15_3.yaml`
- `csgo_configs/exp17.yaml`
- `csgo_configs/exp17_1.yaml`
- `record.md`
- `train_csgo.py`
