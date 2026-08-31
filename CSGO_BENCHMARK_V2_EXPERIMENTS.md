# UniLIP-CS2 Benchmark v2 Baseline Experiments

Status: implementation specification for the `exp31` through `exp34` baseline
families. Training is intentionally not run as part of this change.

## 1. Scope

These experiments move the existing three-map baselines to the frozen
Benchmark v2 protocol without changing their model architecture, trainable
modules, losses, data augmentation, prompt form, or inference post-processing.

The two base-training families are:

| Benchmark v2 experiment | Direct baseline | Task | Training split |
| --- | --- | --- | --- |
| `exp31` | `exp28_1` | joint gen + loc, full-head | Seen-10 train |
| `exp31_loc` | `exp14_3_loc` | loc only, full-head | Seen-10 train |
| `exp31_gen` | `exp14_3_gen` | gen only, full-head | Seen-10 train |
| `exp32` | `exp30_2` | joint gen + loc, LoRA | Seen-10 train |
| `exp32_loc` | `exp14_2_loc` | loc only, LoRA | Seen-10 train |
| `exp32_gen` | `exp14_2_gen` | gen only, LoRA | Seen-10 train |

The CrossMap adaptation families load the corresponding final Seen-10 model:

| Adaptation experiment | Initialization | Task | Training split |
| --- | --- | --- | --- |
| `exp33` | `exp31` | joint gen + loc, full-head | CrossMap-4 support |
| `exp33_loc` | `exp31_loc` | loc only, full-head | CrossMap-4 support |
| `exp33_gen` | `exp31_gen` | gen only, full-head | CrossMap-4 support |
| `exp34` | `exp32` | joint gen + loc, LoRA | CrossMap-4 support |
| `exp34_loc` | `exp32_loc` | loc only, LoRA | CrossMap-4 support |
| `exp34_gen` | `exp32_gen` | gen only, LoRA | CrossMap-4 support |

`exp33*` and `exp34*` are model-weight initialization runs, not Trainer-state
resumes. Optimizer, scheduler, dataloader, and global step start fresh for each
support seed. Their output directories must therefore be unique per shot and
support seed. `finetune_init_ckpt_path` is strict and rejects an initialization
checkpoint missing any key that the adaptation run expects to train.

## 2. Data Contract

Benchmark v2 mode is enabled only when a config declares:

```yaml
benchmark_v2_manifest: "data/csgo_benchmark_v2/benchmark_manifest.json"
benchmark_v2_split: "seen_train"
```

Legacy configs without `benchmark_v2_manifest` retain the existing
`splits_20000_5000` behavior.

The runtime split selectors are:

```text
seen_train
seen_validation
seen_discrete_test
seen_continuous
crossmap_support
crossmap_query_test
crossmap_continuous
```

All consumers must use the manifest's per-map full-corpus `z_min` and `z_max`.
They must not derive Z bounds from Seen train, CrossMap support, query, or
continuous rows. Radar paths also come from the manifest; filename guessing is
not part of Benchmark v2.

Continuous generation reads the explicit ordered clips from
`continuous_clips.json`. Clip IDs and boundaries must survive inference and
evaluation.

## 3. Seen-10 Training

Seen-10 contains 5,000 training frames per map and ten maps, for 50,000 source
frames. The `exp31*` and `exp32*` launch commands retain the corresponding
three-map baseline's GPU count, per-device batch size, gradient accumulation,
epoch count, optimizer settings, scheduler, save interval, image size, short
instruction, and trainability flags.

Because the source-frame count changes from the historical three-map split,
100 epochs no longer imply the historical 46,900 optimizer steps. Benchmark
v2 results must record the actual final global step and checkpoint path.

Seen validation is the only legal Benchmark v2 split for base-model checkpoint
selection or hyperparameter tuning. Seen discrete and continuous test splits
are report-only.

## 4. CrossMap Few-Shot Adaptation

The primary protocol uses 100 support frames per CrossMap map, 400 source
frames in total, and all five frozen support draws (`seed_0` through `seed_4`).
Every seed starts again from the same final Seen-10 checkpoint. A seed must not
start from another seed's adapted checkpoint.

The support selector accepts `benchmark_v2_shots_per_map` and
`benchmark_v2_support_seed` from either YAML defaults or command-line
overrides. The 100-frame support manifest is the immutable parent set. Future
50/20/10-shot subsets are deterministic and nested within that parent for the
same support seed:

```text
10-shot subset of 20-shot subset of 50-shot subset of 100-shot
```

Use a fixed optimizer-step budget when comparing shot counts so that a smaller
support set does not also receive fewer updates. The initial executable recipe
uses 400 optimizer steps at effective source batch 128 with `drop_last=False`.
Each 400-row support epoch is `[128, 128, 128, 16]`, or four updates; this is
approximately 100 complete support-set passes. The naive
`128 * 400 / 400 = 128` calculation is wrong because it treats the final
16-row batch as a full batch. This adaptation budget is provisional: tune it
only through simulated episodes constructed from Seen-10 train/validation,
then freeze it before inspecting CrossMap query results.

The joint adaptation configs continue from the mature Seen loss regime:

- `exp33`: `alpha_loc_loss=20`, `alpha_loc_aux_loss=2`, and
  `alpha_loc_perception_loss=0.1`.
- `exp34`: `alpha_loc_loss=20`; aux-loc and perception remain disabled.

This avoids replaying the base-training warm-up schedules after loading a
fully trained Seen model. All other method fields remain identical to the
direct parent baseline.

## 5. Inference and Evaluation

Seen-10 base models are evaluated on:

- localization: `seen_discrete_test`;
- discrete generation: `seen_discrete_test`;
- continuous generation: `seen_continuous`.

Each adapted CrossMap model is evaluated on the same fixed rows for every
support seed:

- localization: `crossmap_query_test`;
- discrete generation: `crossmap_query_test`;
- continuous generation: `crossmap_continuous`.

The CrossMap query and continuous splits must never be used for early stopping,
checkpoint selection, learning-rate selection, or adaptation-budget tuning.

Report all metrics per map and with an equal-map macro average. For each
CrossMap shot count, aggregate the five support seeds as mean and a two-sided
95% Student-t confidence interval. Support-selection variation and independent
model-training-seed variation are different uncertainty sources and must be
reported separately.

Generation evaluation is strict: its coverage denominator is the selected
manifest rows, not every image present in a raw map directory. Continuous
metrics use the exact Benchmark v2 clips rather than tracks inferred from
filenames. The manifest's 64-frame boundaries define the sequence tracks;
FVD uses 16-frame, stride-16 windows inside each exact 64-frame manifest clip;
it never forms a window across two manifest clips.

Every adapted CrossMap query and Seen-retention inference command must pass
`--benchmark_v2_support_seed "$SUPPORT_SEED"` and
`--benchmark_v2_shots_per_map "$SHOTS"` so the output manifest records the
actual support draw. Keep inference `--seed 42` fixed and independent from the
support seed.

## 6. Output Layout

The documented commands use stable paths:

```text
outputs/csgo_1b/exp31*/model.safetensors
outputs/csgo_1b/exp32*/model.safetensors
outputs/csgo_1b/exp33*/shot_<N>/seed_<S>/model.safetensors
outputs/csgo_1b/exp34*/shot_<N>/seed_<S>/model.safetensors

outputs_eval/benchmark_v2/<experiment>/seen/{discrete,continuous}/
outputs_eval/benchmark_v2/<experiment>/shot_<N>/seed_<S>/{discrete,continuous}/
outputs_loc/benchmark_v2/<experiment>/seen/
outputs_loc/benchmark_v2/<experiment>/shot_<N>/seed_<S>/
```

Exact launch and evaluation commands are maintained in `record.md`.

## 7. Migration Bundle

For a reproducible runtime migration, copy the formal runtime bundle and the
selected source assets. These files are required for the v2 training,
inference, and metric-evaluation commands:

- `data/csgo_benchmark_v2/benchmark_manifest.json`, `calibration/`,
  `splits/`, `aggregate/`, `selected_images.sha256`, `checksums.sha256`, and
  `build_report.json`;
- both released calibration files, `calibration/z_calibration.json` and
  `calibration/z_extrema_rows.jsonl`, must remain available as release and
  verification metadata; only `calibration/*.template.yaml` approval
  templates are outside the runtime bundle;
- the selected FPV source images identified by the manifest and
  `selected_images.sha256`, together with the manifest-referenced radar assets
  under `data/preprocessed_data/` and their original source paths;
- runtime code and tests: `train_csgo.py`, `eval_csgo.py`,
  `eval_csgo_loc.py`, `benchmark_csgo_v1.py`, `benchmark_csgo_v1_conti.py`,
  `csgo_datasets/benchmark_v2.py`, `csgo_datasets/unified_task_dataset.py`,
  `scripts/aggregate_csgo_benchmark_v2_metrics.py`, the relevant `tests/`
  files, and `requirements.txt` (including the PyYAML dependency);
- v2 configs and documentation: `csgo_configs/exp31*.yaml`,
  `csgo_configs/exp32*.yaml`, `csgo_configs/exp33*.yaml`,
  `csgo_configs/exp34*.yaml`, their test configs, `record.md`, and the three
  v2 protocol/experiment documents.

Do not migrate the build-only audit workspace for ordinary training,
inference, or metric evaluation. This includes `data/csgo_benchmark_v2/audit/`
and `data/csgo_benchmark_v2/audit_archive/`, their candidate CSV/JSONL files,
audit reports, review exports, neighboring-frame context images,
extrema/jump visualizations, PDF reports, and other generated intermediate
files. The approval templates
under `data/csgo_benchmark_v2/calibration/*.template.yaml` are also
build-only. These paths are intended to be excluded by `.gitignore` and are
needed only when re-running audit/review or reconstructing/rebuilding the
benchmark. Their omission from a runtime migration must not be extended to
the formal manifest, checksum, aggregate, calibration JSONL/JSON, split, or
selected source-asset files listed above.

The destination still needs the external model and runtime prerequisites used
by the parent experiments, including `UniLIP-1B`,
`OpenGVLab/InternVL3-1B-hf`, compatible CUDA/PyTorch/DeepSpeed dependencies,
and the external localization/FVD dependencies and checkpoints. The full
approximately 17 GB audit workspace is needed only to reconstruct or re-audit
the benchmark; it is not needed once the released runtime bundle and source
assets have been copied and their checksums verified.
