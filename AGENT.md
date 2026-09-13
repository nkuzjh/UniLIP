# UniLIP-CSGO Benchmark Project Entry

Snapshot date: 2026-08-31

Scope: This document is the high-signal project entry for the UniLIP-CSGO
localization and generation benchmark in this checkout. It records the task,
data contract, implementation map, evidence boundary, current experiments,
and next operational priorities. It is not a replacement for the general
UniLIP README, TRAIN.md, or EVAL.md workflows.

## Evidence Levels

Use the following evidence levels for every experiment claim:

- Implemented: code, configuration, or an evaluation path exists and can be
  located in the repository.
- Trained: a concrete output directory contains a checkpoint or trainer state
  for the named experiment and step.
- Evaluated: an evaluation script produced metrics or artifacts for the named
  checkpoint, with the relevant command or output recorded.
- Conclusion: a scientific claim supported by a matched protocol, controlled
  comparison, and an appropriate amount of repeated evidence.

The existence of a YAML file is only Implemented evidence. It is never enough
to claim that an experiment was started, trained, evaluated, or successful.
Similarly, a checkpoint without a traceable evaluation artifact is not
Evaluated evidence. A single run is useful evidence, but is not by itself a
robust general conclusion.

## Project Question

UniLIP-CSGO extends UniLIP with a paired first-person-view and radar setting.
The benchmark tests whether one model can learn both camera-pose localization
and view generation, including generation at continuous unseen frames.

The central research questions are:

1. Does joint localization and generation training improve the two tasks over
   matched single-task controls?
2. Do auxiliary pose and perception objectives improve generation without
   sacrificing localization?
3. Can a model trained on discrete frame pairs generate coherent continuous
   unseen-frame sequences?

The current evidence must distinguish the joint-training effect from the
effects of auxiliary losses, LoRA parameterization, map count, training step
count, checkpoint selection, and evaluation protocol.

## Tasks and Data

The two benchmark directions are:

- loc: FPS image + radar/map image -> normalized 5DoF pose
  `[x, y, z, pitch, yaw]`.
- gen: radar/map image + 5DoF pose -> FPS image.

The current experiment history primarily uses `de_dust2`, `de_nuke`, and
`de_ancient`. Each of these legacy benchmark maps has:

- 20,000 training samples in `splits_20000_5000/train_split.json`.
- 5,000 discrete test samples in `splits_20000_5000/test_split.json`.
- A continuous unseen-frame split in
  `splits_20000_5000/continuous_unseen_clips.json`.

The current continuous unseen counts are:

| Map | Continuous unseen frames |
| --- | ---: |
| `de_dust2` | 2,164 |
| `de_nuke` | 4,602 |
| `de_ancient` | 4,098 |

The train split and the continuous unseen split have zero `file_frame`
overlap under the current identity check. The continuous split overlaps the
discrete test split by 67 Dust2, 165 Nuke, and 105 Ancient `file_frame`
entries; disclose both facts when describing continuous generalization.

Pose normalization is `x / 1024`, `y / 1024`, per-map training-range
normalization for `z`, and angle division by `2*pi` for pitch/yaw. Evaluation
must use the training split as the `z`-range reference.

These legacy splits are frame-random after interval sampling. A repository
audit found that train and discrete test contain all of the same `file_num`
records on every one of the 14 maps. The old continuous split excludes exact
training frames but reuses training records and overlaps the discrete test by
some frames. It is useful for experiment continuity, but it is not a valid
trajectory-generalization protocol.

Benchmark v2 is the next formal protocol. Its complete contract and manual
build procedure are in
[`CSGO_BENCHMARK_V2_CONSTRUCTION.md`](CSGO_BENCHMARK_V2_CONSTRUCTION.md). The
high-level design is:

- Seen-10: `cs_agency`, `cs_italy`, `de_ancient`, `de_anubis`, `de_dust2`,
  `de_inferno`, `de_mirage`, `de_nuke`, `de_overpass`, and `de_train`.
- CrossMap-4: `cs_office`, `de_golden`, `de_palacio`, and `de_vertigo`.
- Seen per map: 5,000 train, 500 validation, 2,000 discrete test, and 20 x
  64-frame continuous clips.
- CrossMap per map: a fixed 2,000-frame query, five built 100-shot support
  draws, and 20 x 64-frame continuous clips. Current `exp33*`/`exp34*` runs use
  only the default draw `seed_0`; the others remain available for optional
  future analysis.
- All pools are disjoint by parsed `file_num`; continuous clip boundaries are
  preserved. Until capture-session IDs are available, claims must say
  `file_num` trajectory-disjoint rather than session-disjoint.
- Coordinate anomalies require an audit fingerprint and explicit human
  approval. The builder never silently clips, repairs, or removes them.
- Each map uses exact `z_min`/`z_max` from every retained row in the approved
  full capture corpus. This calibration is generated and approved before any
  record is assigned to Seen or CrossMap pools, and all splits/support seeds
  share the same frozen values.

The v2 configuration is `csgo_configs/benchmark_v2.yaml`, the approved anomaly
decision is `csgo_configs/benchmark_v2_anomaly_decisions.yaml`, and the builder
is `scripts/build_csgo_benchmark_v2.py`. The formal build is complete under
`data/csgo_benchmark_v2`. The formal runtime bundle is
`benchmark_manifest.json`, `build_report.json`, `checksums.sha256`,
`selected_images.sha256`, `aggregate/`, `calibration/`, and `splits/`, plus
`minimal_dataset_report.json`, the selected flat FPV images under
`images/<map>/`, and the copied manifest-referenced radar files under
`radars/<map>/`. The original `data/preprocessed_data/` paths are not part of
the migration bundle. The
released `calibration/z_calibration.json` and
`calibration/z_extrema_rows.jsonl` are required release/verification metadata
and must be retained; only `calibration/*.template.yaml` approval templates
are build-only. The release checksums currently pass independent
verification.

The `audit/` and `audit_archive/` trees contain audit reports, candidate
tables, review exports, context/extrema/jump images, PDF visualizations, and
historical snapshots. They are build-only workspace artifacts intended to be
excluded by `.gitignore`, and are needed only to re-run audit/review or
reconstruct/rebuild the benchmark. They are not required for training,
inference, or metric evaluation after the formal runtime bundle and source
assets have been verified; this does not make any formal runtime artifact
disposable.

The v2 consumer path is implemented as an explicit opt-in. Training datasets
read the manifest-selected rows, frozen full-corpus per-map Z ranges and
manifest radar paths; training accepts support-seed/shot overrides and
model-only `finetune_init_ckpt_path`. That initialization is strict and rejects
checkpoints missing any adaptation-trainable key. Generation/localization inference accepts
deterministic output/checkpoint/seed overrides plus v2 split/map/support-seed/
shot overrides, and the v2 metric scripts use manifest-selected coverage,
frozen Z, and exact continuous clips. Legacy
configs without `benchmark_v2_manifest` retain their existing behavior.
Benchmark v2 model execution is in progress overall. The localization branch
has completed the Seen-10 baselines and the full nested 100/50/20/10-shot
`exp33_loc`/`exp34_loc` adaptation curves, including all CrossMap-4 and
Seen-retention summaries. Current empirical status and equal-map macro results
are recorded in
[`csgo_benchmark_v2_experiments_results.md`](csgo_benchmark_v2_experiments_results.md).

### Benchmark v2 Runtime and Experiments

The runtime split selectors are `seen_train`, `seen_validation`,
`seen_discrete_test`, `seen_continuous`, `crossmap_support`,
`crossmap_query_test`, and `crossmap_continuous`. All v2 consumers use the
manifest's per-map calibration and explicit radar mapping. Continuous consumers
preserve clip ID and frame order; generation metric coverage is restricted to
the selected manifest rows.

| Seen-10 experiment | Parent | CrossMap-4 adaptation |
| --- | --- | --- |
| `exp31` | `exp28_1`, joint full-head | `exp33` |
| `exp31_1` | `exp31`, strict loss-only ablation without aux-loc/perception | none in the current protocol |
| `exp31_loc` | `exp14_3_loc`, loc full-head | `exp33_loc` |
| `exp31_gen` | `exp14_3_gen`, gen full-head | `exp33_gen` |
| `exp32` | `exp30_2`, joint LoRA | `exp34` |
| `exp32_loc` | `exp14_2_loc`, loc LoRA | `exp34_loc` |
| `exp32_gen` | `exp14_2_gen`, gen LoRA | `exp34_gen` |
| `exp35_<map>` / `exp35_gen_<map>` / `exp35_loc_<map>` | matching `exp31*` Seen-10 checkpoint | map-specific CrossMap-4 full-head adaptation |
| `exp36_<map>` / `exp36_gen_<map>` / `exp36_loc_<map>` | matching `exp32*` Seen-10 checkpoint | map-specific CrossMap-4 LoRA adaptation |

`exp31_1` keeps the `exp31` balanced joint sampling (`task_mix_ratio=0.5`),
full-head trainability, frozen LLM, main localization-loss schedule, effective
per-module learning rates, and launch hyperparameters. Its only method change
is disabling auxiliary localization and localization perception losses, so
`exp31` versus `exp31_1` isolates those losses without mixing in the
LoRA/full-head difference carried by `exp32`.

The Seen runs use 50,000 train frames, 5,000 validation frames, 20,000
discrete-test frames and 200 exact continuous clips. Each adaptation point
starts independently from its matching final Seen model and runs once with the
default frozen support draw (`seed_0`). The localization curve uses nested
100/50/20/10-shot subsets and the provisional fixed budget
`MAX_STEPS=400`. The task-homogeneous sampler emits only complete batches. The
completed 100-shot run used batch 128 (three batches per epoch); the reduced
50/20/10-shot runs use batches 128/80/40 so every dataset has at least one
batch. The protocol therefore fixes optimizer updates, not examples per update
or total example presentations.

The machine-readable localization matrix is
`csgo_configs/benchmark_v2_loc_few_shot.yaml`. Its runner,
`scripts/run_csgo_benchmark_v2_loc_few_shot.py`, exposes `validate`, `status`,
single-pipeline `run`, and resource-gated `schedule` commands. The scheduler
does not launch reduced-shot jobs until both 100-shot CrossMap and Seen
retention summaries pass strict provenance checks. Independent jobs may share
one GPU only when the configured free-memory threshold allows it; training and
the two inference/metric stages within a job remain serial.

Generation and localization metric JSONs are aggregated with
`scripts/aggregate_csgo_benchmark_v2_metrics.py`. The `maps` subcommand uses
`--manifest --split --input_root --kind --output`; it derives the exact map list
from the manifest and reports an equal-map macro, so it does not accept the
legacy-style `--input-dir`, `--protocol`, or `--maps` options. Strict metric
aggregation requires complete selected coverage and each inference output
root's `inference_manifest.json`; do not use the debug override flags for
reported results. Keep inference `--seed 42` fixed and pass support seed `0`
independently as `--benchmark_v2_support_seed 0`. Localization results are
written as strict `benchmark_csgo_v2_loc.json` summaries with an equal-map
macro and provenance. The generic `seeds` subcommand remains implemented for
optional future multi-draw analysis, but it is not used by the single-draw
`exp33*`/`exp34*` protocol and no support-selection confidence interval is
reported.

The map-specific few-shot line targets `cs_office`, `de_golden`, `de_palacio`,
and `de_vertigo`. Each `<map>` run trains only that map's
`crossmap_support` rows with `support_seed=0`, `SHOTS=100` by default
(parameterized for 50/20/10), and `MAX_STEPS=400`. Every map, shot, and task
starts independently from its matching Seen-10 checkpoint; `exp35*` never
initializes from `exp33*`, and `exp36*` never initializes from `exp34*`.
Joint adaptation keeps the exp33/exp34 batch contract `4/4/32`; single-task
adaptation sets `BATCH_SIZE=SHOTS` for 100/50/20/10 because the one-map
support set would otherwise be smaller than the sampler's batch and be
dropped as an incomplete task-homogeneous batch. Query inference is restricted to
the target map, while Seen-retention inference covers all Seen-10 maps with
inference seed `42`. The detailed matrix and command contract are in
[`CSGO_BENCHMARK_V2_MAP_SPECIFIC_FEWSHOT.md`](CSGO_BENCHMARK_V2_MAP_SPECIFIC_FEWSHOT.md).

For `is_multi_task_balanced: True`,
`UniLIPMultiTaskBalancedDataset` stores one base data entry per frame and
returns `[loc_sample, gen_sample]` from `__getitem__`. The collator flattens
these lists before building the batch. Thus one base sample supplies both task
instances to the model, while the dataset length remains the number of base
samples. This is different from drawing one task by a random mix ratio.

## Repository Map

The CSGO training and evaluation path is organized as follows:

- `train_csgo.py`: argument parsing, YAML overrides, model construction,
  dataset selection, optimizer setup, callbacks, checkpointing, and the
  training entry point.
- `csgo_datasets/unified_task_dataset.py`: CSGO sample construction,
  normalized pose handling, task-specific image routing, balanced dataset,
  and collator flattening.
- `unilip/model/language_model/unified_unilip.py`: shared UniLIP model,
  generation branch, localization branch, loss routing, auxiliary losses,
  perception alignment, and trainable-module gating.
- `unilip/train/nonmix_trainer.py`: `NonMixTrainer`, custom optimizer groups,
  logging of model loss diagnostics, and the Hugging Face Trainer loop.
- `eval_csgo.py`: generation inference over either the discrete test split or
  the continuous split; metric computation is a separate benchmark step.
- `eval_csgo_loc.py`: localization inference and normalized/physical pose
  metrics.
- `benchmark_csgo_v1.py`: paired image benchmark metrics and coverage checks.
- `benchmark_csgo_v1_conti.py`: continuous-track metrics, temporal metrics,
  and FVD evaluation.
- `CSGO_BENCHMARK_METRICS_ZH.md`: Chinese reporting-metric selection and
  field-by-field definitions for the localization, discrete-generation, and
  continuous-generation result JSONs.
- `scripts/aggregate_csgo_benchmark_v2_metrics.py`: equal-map and
  support-selection aggregation for strict v2 generation and localization
  metric JSONs.
- `CSGO_BENCHMARK_V2_CONSTRUCTION.md`: formal Seen-10/CrossMap-4 construction
  protocol, anomaly review, output contract, and acceptance criteria.
- `CSGO_BENCHMARK_V2_TASK_SETTING.md`: Benchmark v2 baseline inheritance,
  training/evaluation protocol, and output layout.
- `CSGO_BENCHMARK_V2_MAP_SPECIFIC_FEWSHOT.md`: map-specific CrossMap-4
  few-shot matrix, naming contract, fairness caveat, and aggregation commands.
- `csgo_benchmark_v2_experiments_results.md`: current execution status and
  equal-map macro result tables.
- `scripts/build_csgo_benchmark_v2.py`: deterministic `audit`, `calibrate`,
  `build`, and `validate` commands for benchmark v2.
- `scripts/export_csgo_benchmark_v2_review.py`: read-only CSV export for the
  current candidate list and provisional tied Z extrema.
- `csgo_configs/benchmark_v2.yaml`: frozen map, count, record-pool, sampling,
  audit, and full-corpus calibration settings for v2.
- `record.md`: training commands, checkpoint steps, evaluation commands, and
  output references.
- `csgo_configs/AGENT.md`: detailed experiment history, configuration design,
  module learning rates, and config-specific notes.

The normal CSGO training runs use no inline evaluation. The main commands set
`eval_strategy: no`; evaluation is a separate operation using the checkpoint
and the corresponding test configuration. Do not treat a successful training
process as an evaluation result.

Canonical evaluation outputs are:

- Localization: normalized-space losses plus physical XY, Z, pitch, and yaw
  errors from `eval_csgo_loc.py`.
- Discrete generation: coverage, PSNR, SSIM, Boundary F1, LPIPS, pixel error,
  external-locator pose error, FID, IS, CLIP, and Aesthetic from
  `benchmark_csgo_v1.py`.
- Continuous generation: the discrete metrics plus sequence metrics, temporal
  warping/difference, flicker, optical-flow EPE, and FVD from
  `benchmark_csgo_v1_conti.py`. FVD uses 16-frame stride-16 windows inside
  each exact 64-frame manifest clip; windows never cross clip boundaries.

Treat the `*_v1.py` benchmark scripts as canonical. Older `benchmark_csgo.py`
and `benchmark_csgo_video.py` paths are legacy unless an experiment explicitly
records their use. Use [`CSGO_BENCHMARK_METRICS_ZH.md`](CSGO_BENCHMARK_METRICS_ZH.md)
for the formal main-table/appendix metric set and the Chinese definitions of
every metric retained in current or historical result JSONs.

## Model Branches

The shared language model produces contextual features for both task routes.
The current full-head `exp28_1` family fixes the vision tower and, in its
recorded launch command, fixes the shared language model. It trains the task
heads and connectors according to the config. The LoRA family uses explicit
gating and separate optimizer groups.

Generation branch:

- `llm_connector` processes the generation context.
- `projector` maps context features to the SANA DiT caption channel.
- `latent_queries` provide the generation query sequence.
- SANA `dit` predicts the image flow/noise target.

Localization branch:

- The Pi05-compatible `action_dit` is the main flow-matching locator when
  `use_pi05_action_dit: True`.
- `action_dit_connector`, `action_dit_projector`, and `action_dit_norm`
  connect language features to the action model.
- `action_in_proj`, `action_out_proj`, `time_mlp_in`, and `time_mlp_out`
  encode the noisy pose/time and decode the predicted velocity.
- Optional localization modules include `loc_learnable_query`,
  `regression_loc_head`, `cross_view_fusion`, and `vit_loc_fusion`.

The generation and localization branches receive task masks. A localization
sample contributes to the localization main loss, and a generation sample
contributes to the generation main loss. A generation sample can also carry
the ground-truth pose and generated-image localization context for auxiliary
losses.

## Loss Contract

The active model objective is assembled in
`unilip/model/language_model/unified_unilip.py` as:

```text
L = L_gen
  + alpha_loc * L_loc
  + alpha_repa * L_repa
  + alpha_loc_repa * L_loc_repa
  + alpha_loc_perception * L_loc_perception
  + alpha_loc_aux * L_loc_aux
  + alpha_gen_aux * L_gen_aux
```

Inactive terms are zero. The task masks are applied inside the branch losses;
the total objective does not imply that every term is computed for every
sample.

Generation main loss:

```text
z_sigma = (1 - sigma) * z + sigma * epsilon
v_target = epsilon - z
L_gen = mean((v_pred(z_sigma, context) - v_target)^2)
```

The implementation computes per-sample SANA DiT MSE over latent dimensions and
then applies the generation loss mask.

Localization main loss:

```text
x_t = t * noise + (1 - t) * pose
u_t = noise - pose
L_loc = mean((v_pred(x_t, context) - u_t)^2)
```

The implementation computes the Pi05 Action DiT velocity MSE over the action
dimensions and applies the localization loss mask. `alpha_loc` is scheduled
in the main `exp28_1` family as `[2, 5, 10, 20]` at steps
`[0, 10000, 18000, 28000]`.

Auxiliary localization loss:

- It decodes the generated flow sample into a predicted FPS image.
- It evaluates localization candidates on that generated image while the
  auxiliary locator modules are frozen for the consistency measurement.
- The combined branch uses two candidates, soft EM responsibilities from
  candidate weighted MSE, and an uncertainty weight from candidate residual
  disagreement.
- In `exp28_1`, the timestep weight is `exp(-5 * sigma)`, with no active-mean
  renormalization, and the auxiliary coefficient is scheduled from 0 to 2.

In short, the combined auxiliary term is a weighted per-sample form of:

```text
L_loc_aux = uncertainty_weight * sum_i(q_i * weighted_MSE_i)
```

where `q_i` is the EM responsibility and `weighted_MSE_i` includes the
configured timestep weight.

Localization perception loss:

- The student is the generated FPS image and the teacher is the real target
  FPS image.
- The current mainline feature source is `vision_tower`.
- The feature loss is Smooth L1 per feature dimension, averaged over patch
  tokens, weighted by `(1 - sigma)`, and masked to generation samples.
- With attention weighting enabled, the teacher-ground-truth localization
  action trace reads the last Action DiT layer's attention from the action token
  to the localization-side FPS/understanding patch tokens. Heads are reduced
  by mean and patch weights are
  normalized to mean one and detached.

The current perception schedule is `[0, 0, 0.1]` at steps
`[0, 1999, 2000]`. The implementation names the teacher type
`current_loc_head`; EMA teacher mode is not active in the current path.

Implemented but not all active in the current mainline:

- REPA and localization REPA feature alignment.
- `aux_gen_loss`, which trains a generation consistency route from predicted
  pose information.
- Token localization routes, including `lm_bin_ce` and
  `lm_st_ext_vocab` with LAPE/NTP-style handling.
- LoRA injection for the shared language model, LLM connector, SANA DiT, and
  Action DiT, plus full-train projector and adapter modules.

These are implementation capabilities, not automatic claims about the loss
composition of every experiment. The active YAML and recorded launch command
must be checked for each run.

## Experiment Status

The following is the project status at the snapshot date.

| Experiment | Configuration and evidence | Current interpretation |
| --- | --- | --- |
| `exp28_1_dust2` | Full-head balanced joint training with exp-sigma combined aux_loc and attention-weighted vision perception; `15700/15700` complete; discrete gen, continuous gen, and loc evaluated. | Current strongest comprehensive Dust2 candidate under the recorded single-run protocol. |
| `exp14_3_dust2_gen` / `exp14_3_dust2_loc` | Full-head single-task controls; trained and evaluated. | Matched Dust2 gen/loc controls for the full-head family. |
| `exp28_1_1_dust2` | Joint-only ablation; `12000/15700`; no evaluation. | Aux_loc and perception are both disabled. |
| `exp28_1_2_dust2` | Aux-only ablation; not started; no checkpoint. | Combined aux_loc enabled, perception disabled. |
| `exp28_1_3_dust2` | Perception-only ablation; `4000/15700`; no evaluation. | Perception enabled, aux_loc disabled. |
| `exp28_1` | Three-map joint run; `16000/46900`; no evaluation. | Step 16000 is close to Dust2 step 15700; evaluate it first as a matched-step comparison. |
| `exp14_3_gen` / `exp14_3_loc` | Three-map single-task controls; trained and evaluated. | Three-map full-head controls. Do not compare 46.9k and 15.7k steps directly. |
| `exp30_2` | Three-map pure-LoRA joint control; trained and evaluated. | Three-map LoRA control for the LoRA matrix. |
| `exp30_dust2` / `exp30_1_dust2` and Dust2 `exp14_2` LoRA matrix | Trained and evaluated. | Generation changes are small, while localization degrades; results do not support a shared-LoRA joint-overall-better conclusion. |
| `exp29_dust2` / `exp29_1_dust2` | Token-localization variants; trained and evaluated. | Useful token-route evidence, but not the current best overall route. |

Status terms above follow the Evidence Levels section. In particular, the
three new factor ablations are not complete merely because their configs and
partial checkpoints exist.

## Checkpoint-12000 Evidence

The following values are the recorded single-run Dust2 checkpoint-12000
comparison. Generation rows use `exp14_3_dust2_gen`; the localization row uses
`exp14_3_dust2_loc`.

| Metric | `exp28_1_dust2` | Matching single-task control |
| --- | ---: | ---: |
| Discrete generation PSNR | 17.810 | 17.492 |
| Discrete generation LPIPS | 0.393 | 0.410 |
| Discrete generation FID | 14.671 | 14.602 |
| Continuous generation FVD | 336.673 | 359.695 |
| Localization XY | 27.667 | 29.988 |

These are mixed metrics from a single run and are not a complete statistical
study. They suggest that `exp28_1_dust2` is a promising combined candidate,
but they are not sufficient to attribute a gain specifically to aux_loc,
perception, or joint training. The three factor ablations and matched-step
evaluation are required for that attribution.

## Risks and Next Priorities

1. Run the Seen-10 `exp31*` (including `exp31_1`) and `exp32*`
   training/evaluation matrix, recording
   actual final optimizer steps and checkpoint paths. Use Seen validation only
   for checkpoint and adaptation-recipe decisions.
2. Run CrossMap-4 adaptation for `exp33*` and `exp34*` once with default
   support seed `0`, then report per-map results and equal-map macro averages.
   The 400-step recipe is provisional and must be tuned on simulated Seen
   episodes, never on CrossMap query data.
3. Run CrossMap zero-shot inference for the Seen checkpoints with the explicit
   `crossmap_query_test`/`crossmap_continuous` and CrossMap map-list overrides
   recorded in `record.md`; adapted retention can use the analogous
   `seen_discrete_test`/`seen_continuous` overrides.

The current metadata exposes `file_num`, not capture-session identity. Public
claims must therefore remain `file_num` trajectory-disjoint until sessions are
reconstructed. Exact game/map versions and external-localizer training scope
also need to be archived with reported results. The formal v2 build is ready,
but model-level benchmark evidence and independent training-seed replication
are still outstanding.

## Operational Notes

`TrainingArguments` inherits the Hugging Face Trainer arguments and supports
the CLI option `--max_steps`. When `--max_steps > 0`, Hugging Face Trainer
prioritizes it over the epoch count.

The current `train_csgo.py` does not map a YAML `max_steps` key into
`training_args`. Therefore, to match 15,700 optimizer steps, use the CLI,
for example `--max_steps 15700`; do not put `max_steps` only in YAML and
assume it is applied. Keep an epoch value as a fallback if desired, but the
positive CLI value is the controlling setting.

For three-map runs, use matched optimizer steps when comparing against a
Dust2 run. Dataset size changes the number of updates per epoch, so matching
the nominal epoch count is not a valid matched-compute protocol.

`train_csgo.py` currently overwrites `training_args.deepspeed` with
`deepspeed_scripts/zero0.json`. Do not assume that a different CLI
`--deepspeed` value remains effective without checking the source and runtime
log.

The detailed commands belong in `record.md`. The detailed experiment history,
per-module learning rates, and config variants belong in
`csgo_configs/AGENT.md`. Do not duplicate large command blocks or the full
experiment history in this root entry document.

## Scientific Workflow

Use this compact workflow for new benchmark work:

1. Define the hypothesis, baseline, changed factor, fixed factors, and target
   checkpoint steps.
2. Verify the YAML, data split, model flags, trainable modules, and optimizer
   groups before launching.
3. Train with a concrete output directory and record the actual step count,
   seed, world size, batch settings, and checkpoint path.
4. Evaluate discrete generation, continuous generation, and localization
   separately when the model supports all three routes.
5. Run post-hoc image/video benchmarks on the exact evaluation output and
   preserve coverage and provenance metadata.
6. Compare only matched steps and compatible metrics. Repeat runs when a
   conclusion is important.
7. Update the experiment record and report facts, uncertainties, and
   conclusions separately.

## Agent Work Rules

- Read the relevant code and config before editing.
- Keep changes minimal and within the requested scope.
- Preserve unrelated user changes in a dirty worktree.
- Never use destructive commands such as `git reset --hard` or
  `git checkout --` without explicit authorization.
- Use `rg` for repository search and `apply_patch` for manual edits.
- Prefer structured parsers for YAML, JSON, and checkpoint metadata.
- Do not claim a behavior, result, or fix without execution evidence or a
  clearly labeled source-level inference.
- Keep experiment logic, refactoring, and bug fixing as separate changes.
- Before reporting completion, verify the intended file diff and run the
  smallest relevant validation available.

When handing work to another thread, report the objective, confirmed findings,
files inspected, files modified, remaining work, next step, and risks. The
next engineer should be able to continue from this document, `record.md`, and
`csgo_configs/AGENT.md` without relying on hidden conversation context.
