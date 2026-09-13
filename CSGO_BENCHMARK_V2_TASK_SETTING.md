# UniLIP-CS2 Benchmark v2 Baseline Experiments

Status: implemented protocol for the `exp31` through `exp34` baseline families,
including the `exp31_1` loss-only ablation and the four-arm Seen-3 loss
ablation defined in `CSGO_BENCHMARK_V2_SEEN3_ABLATION.md`.
The `exp33_loc`/`exp34_loc` localization 100/50/20/10-shot curves, including
CrossMap and Seen-retention metrics, completed on 2026-09-04. Other joint and
generation experiments retain their independently tracked status.

## 1. Scope

The main experiment families move the existing three-map baselines to the
frozen Benchmark v2 protocol without changing their model architecture,
trainable modules, losses, data augmentation, prompt form, or inference
post-processing. `exp31_1` is the deliberate exception for loss ablation: it
keeps the `exp31` architecture and trainability fixed while removing two loss
terms. The separate `exp31*_3maps` family repeats the complete `2 x 2`
aux-loc/perception matrix on a fixed three-map subset to reduce experiment
cost; it is not a replacement for the formal Seen-10 protocol.

The two base-training families are:

| Benchmark v2 experiment | Direct baseline | Task | Training split |
| --- | --- | --- | --- |
| `exp31` | `exp28_1` | joint gen + loc, full-head | Seen-10 train |
| `exp31_1` | `exp31` | joint gen + loc, full-head; no aux-loc/perception | Seen-10 train |
| `exp31_3maps` | `exp31` method | joint gen + loc, aux-loc + perception | Seen-3 train |
| `exp31_1_3maps` | `exp31_1` method | joint gen + loc, no aux-loc/perception | Seen-3 train |
| `exp31_2_3maps` | Seen-3 matrix | joint gen + loc, aux-loc only | Seen-3 train |
| `exp31_3_3maps` | Seen-3 matrix | joint gen + loc, perception only | Seen-3 train |
| `exp31_loc` | `exp14_3_loc` | loc only, full-head | Seen-10 train |
| `exp31_gen` | `exp14_3_gen` | gen only, full-head | Seen-10 train |
| `exp32` | `exp30_2` | joint gen + loc, LoRA | Seen-10 train |
| `exp32_loc` | `exp14_2_loc` | loc only, LoRA | Seen-10 train |
| `exp32_gen` | `exp14_2_gen` | gen only, LoRA | Seen-10 train |

`exp31_1` is a strict loss-only control for `exp31`. Both use balanced joint
sampling with `task_mix_ratio: 0.5`, the same main localization-loss schedule,
full training of both task heads, a frozen vision tower and LLM, and the same
effective per-module learning rates, scheduler, batch, epoch, and inference
settings. `exp31_1` sets `is_loc_aux_loss: False` and
`is_loc_perception_loss: False`; it does not define their schedules. It has no
CrossMap few-shot adaptation child in the current protocol.

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
resumes. Optimizer, scheduler, dataloader, and global step start fresh for the
single default support draw (`seed_0`). Output directories retain the
`shot_<N>/seed_0` layout so future shot-count comparisons remain unambiguous.
`finetune_init_ckpt_path` is strict and rejects an initialization checkpoint
missing any key that the adaptation run expects to train.

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
frames. The `exp31*` (including `exp31_1`) and `exp32*` launch commands retain
the corresponding three-map baseline's GPU count, per-device batch size,
gradient accumulation,
epoch count, optimizer settings, scheduler, save interval, image size, short
instruction, and trainability flags.

Because the source-frame count changes from the historical three-map split,
100 epochs no longer imply the historical 46,900 optimizer steps. Benchmark
v2 results must record the actual final global step and checkpoint path.

Seen validation is the only legal Benchmark v2 split for base-model checkpoint
selection or hyperparameter tuning. Seen discrete and continuous test splits
are report-only.

## 3.1 Seen-3 Loss Ablation

The fixed Seen-3 subset is `de_ancient, de_dust2, de_nuke` in manifest order.
Every arm independently starts from `UniLIP-1B`, uses 15,000 source frames,
balanced joint sampling, a frozen vision tower and LLM, full task heads, two
GPUs, per-device batch 4, gradient accumulation 16, and 50 epochs. This gives
an expected 5,900 optimizer steps at effective global batch 128 because the
Trainer rounds the final gradient-accumulation group up once per epoch.

The main, aux-loc, and perception schedule steps are scaled by 0.3 relative to
Seen-10. The four arms are full (`exp31_3maps`), neither loss
(`exp31_1_3maps`), aux-only (`exp31_2_3maps`), and perception-only
(`exp31_3_3maps`). Inference is restricted to 6,000 Seen-3 localization and
discrete-generation rows plus 3,840 Seen-3 continuous-generation rows. There
is no CrossMap evaluation or adaptation in this protocol. Generation metrics
must aggregate the explicit ordered map subset, while localization enables
`benchmark_v2_allow_map_subset_summary: True`.

The complete contract, output layout, and interpretation rules are in
`CSGO_BENCHMARK_V2_SEEN3_ABLATION.md`.

## 4. CrossMap Few-Shot Adaptation

The primary protocol uses the default frozen support draw (`seed_0`). The
reported localization curve uses 100/50/20/10 support frames per CrossMap map,
or 400/200/80/40 source frames in total. Every point starts independently from
the matching final Seen-10 checkpoint, never from another adapted checkpoint.
The benchmark bundle still contains the other frozen support draws for optional
future analysis, but `exp33*` and `exp34*` do not run them.

The support selector accepts `benchmark_v2_shots_per_map` and
`benchmark_v2_support_seed` from either YAML defaults or command-line
overrides. The 100-frame support manifest is the immutable parent set. Future
50/20/10-shot subsets are deterministic and nested within that parent for the
default support draw:

```text
10-shot subset of 20-shot subset of 50-shot subset of 100-shot
```

Use a fixed 400 optimizer-step budget when comparing shot counts so that a
smaller support set does not also receive fewer updates. The current
`DistributedTaskTypeBatchSampler` emits only complete task-homogeneous batches,
even when `dataloader_drop_last=False`; this flag controls distributed
batch-list padding, not the tail inside a task group. The completed 100-shot
run used batch 128, yielding three batches per epoch and dropping a shuffled
16-row tail. The runner keeps batch 128 for 50-shot (one batch and a shuffled
72-row tail), then uses batches 80 and 40 for 20-shot and 10-shot so those
datasets yield one complete batch. The comparison fixes update count, not
examples per update or total example presentations. This adaptation budget is
provisional: tune it only through simulated episodes constructed from Seen-10
train/validation, then freeze it before inspecting CrossMap query results.

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

Each adapted CrossMap model is evaluated once using the default support draw
`seed_0` and the same fixed query rows:

- localization: `crossmap_query_test`;
- discrete generation: `crossmap_query_test`;
- continuous generation: `crossmap_continuous`.

The CrossMap query and continuous splits must never be used for early stopping,
checkpoint selection, learning-rate selection, or adaptation-budget tuning.

Report all metrics per map and with an equal-map macro average. The `exp33*`
and `exp34*` protocol reports one point estimate for support seed `0`; it does
not report a cross-support-seed mean or confidence interval. This single draw
does not measure support-selection or independent model-training uncertainty.
The metric subset used in formal main tables and appendices, together with
Chinese definitions of every result-JSON metric, is maintained in
[`CSGO_BENCHMARK_METRICS_ZH.md`](CSGO_BENCHMARK_METRICS_ZH.md).

Generation evaluation is strict: its coverage denominator is the selected
manifest rows, not every image present in a raw map directory. Continuous
metrics use the exact Benchmark v2 clips rather than tracks inferred from
filenames. The manifest's 64-frame boundaries define the sequence tracks;
FVD uses 16-frame, stride-16 windows inside each exact 64-frame manifest clip;
it never forms a window across two manifest clips.

Every adapted CrossMap query and Seen-retention inference command must pass
`--benchmark_v2_support_seed 0` (or an equivalent `SUPPORT_SEED=0` variable) and
`--benchmark_v2_shots_per_map "$SHOTS"` so the output manifest records the
actual support draw. Keep inference `--seed 42` fixed; this is inference
randomness and is independent from support seed `0`.

## 6. Output Layout

The documented commands use stable paths:

```text
outputs/csgo_1b/exp31*/model.safetensors
outputs/csgo_1b/exp32*/model.safetensors
outputs/csgo_1b/exp33*/shot_<N>/seed_0/model.safetensors
outputs/csgo_1b/exp34*/shot_<N>/seed_0/model.safetensors

outputs_eval/benchmark_v2/<experiment>/seen/{discrete,continuous}/
outputs_eval/benchmark_v2/<experiment>/shot_<N>/seed_0/{discrete,continuous}/
outputs_loc/benchmark_v2/<experiment>/seen/
outputs_loc/benchmark_v2/<experiment>/shot_<N>/seed_0/
```

Exact launch and evaluation commands are maintained in `record.md`.
Current execution status and equal-map macro results are maintained in
[`csgo_benchmark_v2_experiments_results.md`](csgo_benchmark_v2_experiments_results.md).

The localization curve is also encoded in
`csgo_configs/benchmark_v2_loc_few_shot.yaml` and executed by
`scripts/run_csgo_benchmark_v2_loc_few_shot.py`. The runner validates the real
nested support rows and parent checkpoints, requires both 100-shot localization
pipelines to have complete CrossMap and Seen-retention summaries, and then runs
each 50/20/10-shot pipeline in the order train, CrossMap inference plus metrics,
Seen-retention inference plus metrics. It skips only artifacts whose step count,
sample count, split, maps, support metadata, checkpoint and inference provenance
all match the protocol. Its scheduler may run independent pipelines in parallel
subject to the configured free-memory threshold; stages inside one pipeline are
always serial. Launches use a 180-second allocation-settle interval, and the
external 100-shot prerequisite gate has an explicit six-hour timeout.

The current Trainer checkpoint format records final optimizer steps but does
not bind the model to a support-manifest hash or parent-checkpoint hash. Formal
runs therefore use fresh `shot_<N>/seed_0` output directories and retain the
runner's exact stage logs as training provenance. Inference summaries are
checked more strongly: benchmark manifest, split, map order, support metadata,
checkpoint path, sample count, seed and referenced inference manifest must all
match.

## 7. Migration Bundle

Benchmark v2 has one explicit asset-backend switch:
`benchmark_v2_asset_manifest`. When it is absent or `null`, the consumer keeps
the historical source layout `data_dir/<map>/imgs/<file>.jpg` (and the radar
files below `data_dir`), so the existing `exp31`--`exp36` configurations run
unchanged. When it is set to
`data/csgo_benchmark_v2/minimal_dataset_report.json`, training, generation and
localization inference, and discrete/continuous metric runners resolve assets
from the flat `images/<map>/` and `radars/<map>/` trees and ignore `data_dir`.
This is a path-routing choice; it neither moves nor links the source corpus.

The four v2 matrix runners expose this field at matrix top level and accept the
same name as a CLI override:
`run_csgo_benchmark_v2_gen.py`, `run_csgo_benchmark_v2_loc_few_shot.py`,
`run_csgo_benchmark_v2_map_specific.py`, and
`run_csgo_benchmark_v2_checkpoint_eval.py`. Direct train and inference scripts
accept the field in task YAML and as a CLI override; the two metric scripts
accept the same CLI option. A runner derives the independent metric `--gt`
path from the selected asset backend; it must not be hand-written as a source
`data_dir` path. Runner precedence is CLI override > matrix, and a task YAML
must omit the field or agree with the matrix; a conflict fails immediately.
Direct train/inference precedence is CLI > task YAML > source default. There is
no automatic backend detection. Keeping the historical `data_dir` in a task
YAML while minimal mode is enabled is intentional: minimal mode ignores it and
does not treat its mere presence as backend mixing. Source inference/metric
artifacts still cannot be reused as minimal artifacts because provenance is
checked.

The runtime selector validates the asset report, selected checksum, and radar
contents; `verify-target` authenticates every copied JPG as well. The metric
aggregator checks that recorded asset identities agree across inference and
metric artifacts. It is not a replacement for validating the migrated bundle
with `verify-target`.

For a reproducible formal-release migration, copy the formal flat runtime bundle
and the asset report. The execution-minimal consumer path directly opens the
protocol manifest, requested per-map split files, flat FPV images, and requested
map radar assets; `selected_images.sha256` and `minimal_dataset_report.json`
bind the selected set and target paths. The additional artifacts below preserve
release verification, fallback, and provenance and should remain in the normal
server migration bundle. The measured 2026-09-07 materialization, exact byte
sizes, and the distinction between execution-minimal and formal-release-minimal
are recorded in
[`how_to_build_cs2_dataset.md`](how_to_build_cs2_dataset.md#benchmarkv2-最小同步数据包2026-09-07).

The formal migration bundle is:

- `data/csgo_benchmark_v2/benchmark_manifest.json`, `calibration/`,
  `splits/`, `aggregate/`, `selected_images.sha256`, `checksums.sha256`, and
  `build_report.json`;
- both released calibration files, `calibration/z_calibration.json` and
  `calibration/z_extrema_rows.jsonl`, must remain available as release and
  verification metadata; only `calibration/*.template.yaml` approval
  templates are outside the runtime bundle;
- the selected flat FPV images under
  `data/csgo_benchmark_v2/images/<map>/`, the 14 manifest-referenced radar
  files under `data/csgo_benchmark_v2/radars/<map>/`,
  `selected_images.sha256`, and
  `data/csgo_benchmark_v2/minimal_dataset_report.json`; the original
  `data/preprocessed_data/` source paths are not part of this migration bundle;
- runtime code and tests: `train_csgo.py`, `eval_csgo.py`,
  `eval_csgo_loc.py`, `benchmark_csgo_v1.py`, `benchmark_csgo_v1_conti.py`,
  `csgo_datasets/benchmark_v2.py`, `csgo_datasets/unified_task_dataset.py`,
  `scripts/aggregate_csgo_benchmark_v2_metrics.py`,
  all four `scripts/run_csgo_benchmark_v2_*.py` matrix runners, the relevant
  `tests/` files, and `requirements.txt` (including the PyYAML dependency);
- v2 configs and documentation: `csgo_configs/exp31*.yaml`,
  `csgo_configs/exp32*.yaml`, `csgo_configs/exp33*.yaml`,
  `csgo_configs/exp34*.yaml`, `csgo_configs/exp35*.yaml`,
  `csgo_configs/exp36*.yaml`, their test configs,
  the four `csgo_configs/benchmark_v2_*.yaml` matrix configs, `record.md`, and
  the four v2 protocol/experiment documents.

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
selected flat-asset files listed above.

The formal bundle contains 102,878 files and 9,791,753,144 B (9.119 GiB) of
logical content; its current ext4 allocation is 10,007,040,000 B (9.320 GiB).
The execution-layer minimum is 102,865 files and 9,766,923,830 B (9.096 GiB),
including the 10,940,220 B `selected_images.sha256` and 7,886 B asset report.
It contains 102,780 images (9,732,747,719 B) and 14 radars (3,103,923 B).
On a new server with the same repository-relative paths, synchronize this flat
formal bundle plus the code, v2 configs, and experiment checkpoints/model
dependencies. Existing `exp31`--`exp36` source runs retain their old configs;
when only the flat bundle is present, reproduce them or start `exp37` by
changing only `benchmark_v2_asset_manifest` to the report path above. No
directory remapping, symlink, hard link, or move is required.

With the source corpus available, the materializer's full verification remains:

```bash
python scripts/materialize_csgo_benchmark_v2.py verify
```

On a source-free destination, use the target-only verification instead; it
does not access `data/preprocessed_data/`:

```bash
python scripts/materialize_csgo_benchmark_v2.py verify-target
```

For the current formal report, this command also verifies the bound
`data/csgo_benchmark_v2/build_report.json` and
`csgo_configs/benchmark_v2.yaml`. They are included in the recommended formal
migration set (the config is counted with project files, not in the data-size
table). An execution-layer-only copy can run the consumers, but it does not
satisfy this complete release-verification contract.

The destination still needs the external model and runtime prerequisites used
by the parent experiments, including `UniLIP-1B`,
`OpenGVLab/InternVL3-1B-hf`, compatible CUDA/PyTorch/DeepSpeed dependencies,
and the external localization/FVD dependencies and checkpoints. The full
approximately 17 GB audit workspace is needed only to reconstruct or re-audit
the benchmark; it is not needed once the released flat runtime bundle and its
checksums have been verified.

## 8. Map-specific CrossMap Few-shot Adaptation

This is an additional adaptation setting and does not change the active
`exp31`-`exp34` protocol or its status. It evaluates whether specializing one
model to one CrossMap map changes few-shot transfer quality.

| Family | Model names | Seen initialization | CrossMap support |
| --- | --- | --- | --- |
| Full-head | `exp35_<map>`, `exp35_gen_<map>`, `exp35_loc_<map>` | matching `exp31`, `exp31_gen`, `exp31_loc` | only `<map>` |
| LoRA | `exp36_<map>`, `exp36_gen_<map>`, `exp36_loc_<map>` | matching `exp32`, `exp32_gen`, `exp32_loc` | only `<map>` |

Here `<map>` is one of `cs_office`, `de_golden`, `de_palacio`, and
`de_vertigo`. The model is initialized directly from its matching Seen-10
checkpoint, never from `exp33*` or `exp34*`. Every map, task, and shot count
starts independently from that Seen checkpoint. The default is
`support_seed=0`, `SHOTS=100`, and `MAX_STEPS=400`; deterministic nested
`50/20/10`-shot overrides remain supported.

Joint training uses per-device train batch `4`, eval batch `4`, and gradient
accumulation `32`, retaining the corresponding exp33/exp34 flags. Gen-only and
loc-only training set train `BATCH_SIZE=SHOTS` for `100/50/20/10` shots, eval
batch `128`, and accumulation `1`. A one-map support set smaller than batch
128 would otherwise be dropped as an incomplete task-homogeneous batch.

For an adapted checkpoint, CrossMap query inference selects only its target
map; the same checkpoint is also evaluated on all Seen-10 maps for retention.
Generation uses `crossmap_query_test` and `crossmap_continuous`; localization
uses `crossmap_query_test`. Seen retention uses `seen_discrete_test` and
`seen_continuous`. Keep support seed `0` and inference seed `42` separate and
record both in the inference provenance.

The exact per-model training, inference, and metric commands are in
[`record.md`](record.md). The map-specific experiment matrix, expected config
names, output paths, and family-level `map-models` aggregation examples are
in [`CSGO_BENCHMARK_V2_MAP_SPECIFIC_FEWSHOT.md`](CSGO_BENCHMARK_V2_MAP_SPECIFIC_FEWSHOT.md).
