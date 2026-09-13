# Benchmark v2 Seen-3 Loss Ablation

## 1. Purpose

This protocol is a low-cost `2 x 2` loss ablation for the full-head, frozen-LLM
joint generation and localization route. It is separate from the formal
Seen-10 and CrossMap-4 result matrix: all four variants are trained from the
same `UniLIP-1B` initialization and are evaluated only on the same Seen-3 map
subset.

| Experiment | Aux-loc | Loc perception |
|---|---:|---:|
| `exp31_3maps` | on | on |
| `exp31_1_3maps` | off | off |
| `exp31_2_3maps` | on | off |
| `exp31_3_3maps` | off | on |

The suffix mapping intentionally follows the historical Dust2 ablation:
`_2` is aux-only and `_3` is perception-only.

## 2. Frozen Data Contract

The experiments use the existing Benchmark v2 manifest and do not create a
new split. The fixed subset is listed in manifest order:

```text
de_ancient, de_dust2, de_nuke
```

Each map contributes 5,000 `seen_train` frames, 2,000
`seen_discrete_test` frames, and 20 continuous clips containing 1,280 frames.
The resulting protocol sizes are:

```text
train source frames:        15,000
localization/discrete test:  6,000
continuous test:             3,840
```

No CrossMap-4 inference, few-shot adaptation, or Seen-10 retention evaluation
belongs to this ablation.

## 3. Training Contract

All variants use balanced joint training, full task heads, a frozen vision
tower and LLM, and the same optimizer groups as `exp31_1`:

```yaml
is_multi_task: True
is_multi_task_balanced: True
task_mix_ratio: 0.5
is_lora: False
llm_train_mode: "frozen"
fix_vit: True
fix_llm: True
fix_connect: False
fix_dit: False
freeze_gen_head: False
freeze_loc_head: False
```

The launch geometry is fixed for every arm:

```text
world size:                    2 GPUs
per-device train batch:        4
gradient accumulation:        16
effective global batch:        128 source samples
epochs:                        50
expected optimizer steps:      5,900
```

The custom distributed batch sampler yields 1,875 per-rank micro-batches per
epoch. The Trainer rounds `1,875 / 16` up to 118 optimizer steps per epoch, so
50 epochs produce 5,900 optimizer steps.

The Seen-10 step schedules are multiplied by the data ratio
`15,000 / 50,000 = 0.3`, preserving their position relative to epoch progress:

```yaml
alpha_loc_schedule_steps: [0, 3000, 5400, 8400]
alpha_loc_schedule_values: [2.0, 5.0, 10.0, 20.0]

alpha_loc_aux_schedule_steps: [0, 1800]
alpha_loc_aux_schedule_values: [0.0, 2.0]

alpha_loc_perception_schedule_steps: [0, 599, 600]
alpha_loc_perception_schedule_values: [0.0, 0.0, 0.1]
```

A disabled loss must not declare its schedule. All four runs start from the
same base model independently; none is a continuation or checkpoint resume
from another ablation arm.

## 4. Evaluation Contract

Each final checkpoint is evaluated with inference seed 42 on exactly three
tasks:

```text
localization:          seen_discrete_test, 6,000 rows
discrete generation:  seen_discrete_test, 6,000 rows
continuous generation: seen_continuous, 3,840 rows
```

Localization uses `benchmark_v2_allow_map_subset_summary: True`. Generation
metrics are aggregated with the explicit ordered map subset:

```bash
--maps de_ancient de_dust2 de_nuke
```

The reported values are equal-map macros over these three maps. The aggregate
must retain the Benchmark v2 manifest, split, checkpoint, inference seed,
sample count, and exact map-subset provenance.

In `csgo_benchmark_v2_experiments_results.md`, the ablation result chapter uses
a Setext level-one heading (the title followed by `====================`). This
is intentional compatibility for already-running map-specific workers and
must not be converted to an ATX `#` heading while those workers exist.

## 5. Interpretation

Use the four matched arms for the following contrasts:

```text
aux main effect without perception: exp31_2_3maps - exp31_1_3maps
perception main effect without aux:  exp31_3_3maps - exp31_1_3maps
aux effect with perception:          exp31_3maps - exp31_3_3maps
perception effect with aux:           exp31_3maps - exp31_2_3maps
interaction: exp31_3maps - exp31_2_3maps - exp31_3_3maps + exp31_1_3maps
```

Metric directions must be respected when interpreting deltas. These results
support loss attribution on Seen-3 only and must not be presented as a direct
replacement for the formal Seen-10 comparison.
