# UniLIP-CS2 Benchmark v2 Protocol

Status: the formal v2 audit, approved anomaly decisions, full-corpus Z
calibration, calibration approval, deterministic split construction, manifest,
build report, and checksums are complete under `data/csgo_benchmark_v2`. The
production audit contains 10,345 coordinate candidates and no integrity-invalid
rows. `checksums.sha256` currently passes independent verification for the
released generated artifacts.

The train/evaluation consumers are implemented as an explicit opt-in and have
been source-level verified, including manifest row selection, frozen global Z,
explicit radar paths, exact continuous clips, deterministic inference
provenance, and strict manifest-based metric coverage. No model training,
inference, or metric run has been executed for v2 in this checkout.

The earlier 204,390-candidate audit is preserved under
`data/csgo_benchmark_v2/audit_archive/f6a2ad36_pre_z_mad_report_only`. Its
review exports remain under `audit/review_exports`, including 7,229 provisional
tied-extrema rows calculated with the confirmed explicit exclusions.

Benchmark ID: `csgo_benchmark_v2`

Protocol version: `2.0.0`

The production YAML sets `strict_protocol: true`; the builder rejects changed
map lists, quotas, global/support seeds, record-pool ratios, pose sampling,
clip settings, or full-corpus calibration semantics under the same protocol
ID/version. Any such change requires a new protocol version and output root.

## 1. Purpose

Benchmark v2 replaces the legacy frame-random 3-map/14-map splits with three
separable claims:

1. **Seen-10 unseen-trajectory:** train and evaluate on the same ten maps, but
   use disjoint capture records for train, validation, discrete test, and
   continuous test.
2. **CrossMap-4 zero/100-shot adaptation:** keep four maps completely outside
   base training, evaluate zero-shot transfer, then adapt with exactly 100
   labeled frames per held-out map and evaluate on fixed disjoint queries.
3. **Trajectory-disjoint continuous generation:** evaluate fixed-length clips
   from capture records that never contribute discrete training, validation,
   support, or query frames. Clip boundaries remain explicit.

This protocol is designed for localization and conditional first-person view
generation. It does not turn ordinary random held-out frames from a known
capture into an unseen-trajectory claim.

## 2. Map Assignment

### Seen-10

| Map | Main coverage role |
| --- | --- |
| `cs_agency` | modern indoor/outdoor office geometry |
| `cs_italy` | dense Mediterranean streets and mixed lighting |
| `de_ancient` | vegetation, stone texture, and open/covered transitions |
| `de_anubis` | warm desert architecture and water/vertical transitions |
| `de_dust2` | canonical open corridors and high-contrast desert texture |
| `de_inferno` | dense lanes, interiors, and varied texture |
| `de_mirage` | open courtyards and repeated warm architectural motifs |
| `de_nuke` | industrial indoor/outdoor space and strong vertical layering |
| `de_overpass` | large outdoor routes, tunnels, and elevation changes |
| `de_train` | industrial repetition, occlusion, and long sight lines |

The three historical development maps (`de_dust2`, `de_nuke`, and
`de_ancient`) remain in Seen-10. They must not be relabeled as cross-map
generalization after prior model and hyperparameter development used them.

### CrossMap-4

| Map | Held-out domain axis |
| --- | --- |
| `cs_office` | fluorescent indoor lighting, repeated rooms, low-texture surfaces |
| `de_golden` | warm/dusk mining-village and industrial visual style |
| `de_palacio` | bright high-detail palace/courtyard geometry and vegetation |
| `de_vertigo` | construction-site texture, open skyline, and extreme verticality |

This side intentionally mixes legacy/official and community-map domains. The
map files and capture metadata used for release must be versioned because
community map updates can alter geometry, radar transforms, and appearance.
The legacy capture directories `cs_agency_v0`, `cs_italy_v0`, and
`cs_office_v0` are embargoed by the config and are not eligible for v2.

Radar assets are selected by an explicit per-map path and SHA-256, never by a
filename fallback. This matters because Agency/Golden/Palacio use `_tga.png`,
Nuke/Train/Vertigo use blended radar images, and the current `cs_office` radar
is stored under the shared `maps/` directory rather than `cs_office/`.

## 3. Required Sample Counts

### Seen-10

Each map contributes:

| Split | Frames per map | Ten-map total |
| --- | ---: | ---: |
| train | 5,000 | 50,000 |
| validation | 500 | 5,000 |
| discrete test | 2,000 | 20,000 |
| continuous test | 20 clips x 64 frames | 200 clips / 12,800 frames |

Validation is not part of the 5,000-frame training budget. It is the only
split used for checkpoint selection and base-training hyperparameter tuning.

### CrossMap-4

Each map contributes:

| Split | Frames per map | Four-map total |
| --- | ---: | ---: |
| 100-shot support | 100 | 400 per support seed |
| fixed discrete query | 2,000 | 8,000 |
| fixed continuous query | 20 clips x 64 frames | 80 clips / 5,120 frames |

The builder publishes five deterministic support draws (`seed_0` through
`seed_4`) so optional future studies can measure support-selection variance.
The current `exp33*`/`exp34*` protocol uses only the default draw `seed_0`.
The query and continuous sets are fixed across all published draws. Support
sets may overlap one another, but every support record is from a pool disjoint
from every query and continuous record.

The primary 100-shot result adapts one Seen-10 model jointly on the 400
`seed_0` support frames. A per-map adaptation result can be reported as an
optional analysis, but it is not interchangeable with the primary protocol.

## 4. Trajectory Identity and Isolation

The source schema provides `file_frame`, for example
`file_num17_frame_231`. Benchmark v2 parses:

```text
record_id = 17
frame_id = 231
```

and treats the complete `file_num` record as the available trajectory group.
No record may cross split pools:

```text
Seen:     train | validation | discrete_test | continuous
CrossMap: support | query_test | continuous
```

This is materially stronger than the legacy split, where train and test
contained frames from the same records on all 14 maps. It also removes the
legacy continuous/discrete-test frame overlap.

Important limitation: `file_num` is the strongest trajectory identity in the
current preprocessed metadata. If several `file_num` records came from one
uninterrupted capture session, v2 is record-disjoint rather than fully
session-disjoint. Before public release, preserve or reconstruct a
capture-session ID and group all records from one session together. Claims in
the meantime must use the exact term **file_num trajectory-disjoint**.

## 5. Deterministic Split Algorithm

The builder uses the following fixed procedure per map:

1. Load `positions.json`, parse exact frame identities, and verify source
   images, map labels, finite pose fields, duplicate identities, and radar
   presence.
2. Apply only the exclusions recorded in the approved anomaly-decision file.
   Coordinate candidates are never silently deleted.
3. Before assigning any split, compute one exact `z_min`/`z_max` pair per map
   from every retained row in the approved full corpus. Freeze the calibration,
   all tied extrema rows, source fingerprints, and decision fingerprint behind
   a second human approval.
4. Rank whole records with SHA-256 using the global seed and map identity.
   Allocate records to pools with largest-remainder rounding of the configured
   ratios.
5. Sort frames by record and frame index, then thin discrete candidates to a
   minimum five-frame gap within each record.
6. Select exact discrete quotas with deterministic pose-coverage bins over
   `x`, `y`, `z`, yaw, and pitch.
7. For Seen-10, remove discrete-test candidates near the selected train or
   validation pose. For CrossMap-4, construct one fixed query after filtering
   against the union of all five support draws. The default tolerance is
   `8` radar units in X/Y, `8` in Z, and `5` degrees in both angles.
8. Build continuous clips only from the continuous record pool. Each clip has
   exactly 64 ordered frames, adjacent frame-index gaps in `[1, 2]`, no frame
   reuse, and at most one selected clip per record.
9. Fail rather than reduce a quota when eligible data are insufficient.

Canonical ranking serializes `[global_seed, *context_parts]` as ASCII JSON
with sorted object keys, compact separators, and non-finite values forbidden,
then interprets SHA-256 as an unsigned integer. Largest-remainder ties follow
the declared pool order. Pose-quantile ties use record ID, frame ID, and source
row index; the support seed is part of every support ranking context. This
avoids Python's process-randomized `hash()` and makes tie behavior explicit.

Near-pose filtering is a conservative leakage control, not a semantic
distance metric. Its units must be checked against the final per-map radar
calibration. Continuous clips are guaranteed trajectory-disjoint; they are
not guaranteed pose-disjoint from every training frame and must not be
described that way.

The source angles are radians. `angle_h` is yaw, wraps at `2*pi`, has zero at
East, and increases clockwise. `angle_v` is pitch on the current `0..pi`
down-to-up convention. Yaw comparisons use circular distance; pitch
comparisons do not wrap. These conventions and the exact radar transform must
be fingerprinted with the released map calibration.

## 6. Coordinate and Integrity Review

The workflow has a mandatory human gate:

```text
audit -> review captured frames -> approve anomalies -> calibrate ->
review extrema -> approve calibration -> build -> validate
```

`audit` reports two classes of findings:

- **Integrity findings:** malformed identity, duplicate identity, non-finite
  pose, wrong map label, missing source image, or missing radar asset. A build
  cannot silently accept these rows; they must be corrected or explicitly
  excluded.
- **Coordinate candidates:** X/Y outside the provisional `[0, 1024]` radar
  window, large adjacent-frame pose jumps, or angles outside configured ranges.
  These are review triggers, not automatic errors. Per-map Z median/MAD remains
  report-only because it over-flags valid floors on vertically layered maps.

The decision template starts with `status: pending` and
`default_action: undecided`. The released decision file is approved, records
the exact audit/candidate fingerprints, reviewer/date/source references, a
`keep` default, and explicit frame/record exclusions. The builder checks source
and audit hashes so that an approval cannot be reused after data drift.

The earlier read-only inspection found three items that require particular
attention; these observations came from legacy sampled splits, not the final
v2 audit:

- `de_palacio` has five inspected rows from record 100 at `x=-28, y=711`
  (`frame_190`, `198`, `206`, `214`, and `222`); their Z is `-71` or `-69`.
- `de_vertigo` has two inspected record-126 rows at X `1051` and `1072`, with
  Z `182` and `-231` (`frame_464` and `472`).
- Three inspected record-191 rows (`frame_675`, `685`, and `694`) have Z
  `785`, `135`, and `-231`. The inspected Vertigo sample spans approximately
  `z=-231` to `2986`, while its 1st/50th/99th percentiles are approximately
  `2888/2960/2974`.

Negative or greater-than-1024 radar coordinates may be legitimate after map
crop/transformation, and Vertigo has genuine vertical layers. Conversely, the
small low-Z cluster may represent a loading transition, teleport, spectator
state, or capture error. These observations informed the now-approved
record/frame exclusions; the complete decision file, rather than this short
historical summary, is the authoritative list.

Visual spot checks make the transition hypothesis stronger: Palacio
`file_num100_frame_206` shows the player body against nearby geometry, while
frames 175 and 250 return to ordinary scene views. Vertigo
`file_num126_frame_450/472` and `file_num191_frame_694` show a third-person
falling player; record-126 frame 500 is a letterboxed transition view. These
look unsuitable as FPV generation/localization targets even if their physical
coordinates correspond to real out-of-bounds space. Treat this as preliminary
evidence: inspect all audit candidates plus their neighboring captured frames
before choosing candidate-frame or whole-record exclusions.

Pose rows remain in their original units and fields in every manifest. Before
splitting, the calibration stage applies approved exclusions and computes exact
per-map Z extrema from all remaining rows, including future train, support,
query, and test pools. Loaders normalize with the published frozen pair
`(z - z_min) / (z_max - z_min)` and must neither clamp labels nor recompute a
range from any split.

## 7. Adaptation and Evaluation Protocol

Use this order for each model family:

1. Train one base model on the aggregate Seen-10 train set.
2. Select checkpoints and all adaptation hyperparameters using Seen-10
   validation only.
3. Evaluate Seen-10 discrete and continuous tests.
4. Evaluate the unchanged base checkpoint on the fixed CrossMap-4 query and
   continuous sets. This is the zero-shot result.
5. Restore the same base checkpoint, adapt on the default `seed_0` 100-shot
   support set, then evaluate the fixed CrossMap query and continuous sets.
6. For the `exp33_loc`/`exp34_loc` scaling study, additionally restore the same
   Seen-10 parent independently for the nested 50/20/10-shot subsets.
7. Re-evaluate Seen-10 after every adaptation to report retention/forgetting.

Adaptation initialization is strict: `finetune_init_ckpt_path` rejects a
checkpoint missing any key that the adaptation run expects to train. The
initial localization recipe uses 400 updates. The task-homogeneous sampler
keeps only complete batches inside each task group, regardless of the
`dataloader_drop_last=False` argument. The completed 100-shot run used batch
128 and therefore three batches per epoch from 400 rows. Reduced-shot runs use
per-device batches 128/80/40 for 50/20/10-shot: respectively one complete batch
from 200 rows, one from 80, and one from 40. This fixes optimizer-update count,
not batch cardinality or total example presentations.

Do not tune epochs, learning rate, checkpoint choice, or early stopping on a
CrossMap query. Tune an adaptation recipe and shot-count policy through
simulated episodes inside Seen-10, freeze them, and then apply them to the
default CrossMap support draw (`seed_0`).

Report localization and generation separately, per map and as an equal-map
macro average. The current `exp33*`/`exp34*` 100-shot protocol reports the
single `seed_0` point estimate and does not claim support-selection confidence
intervals. Continuous generation must include sequence-level metrics and
retain clip identity rather than treating all frames as one sequence.

## 8. Output Contract

The default output root is `data/csgo_benchmark_v2`:

```text
audit/
  audit_report.json
  coordinate_candidates.jsonl
  coordinate_candidates.csv
  anomaly_decisions.template.yaml
calibration/
  z_calibration.json
  z_extrema_rows.jsonl
  z_calibration_approval.template.yaml
splits/
  seen/<map>/train.json
  seen/<map>/validation.json
  seen/<map>/discrete_test.json
  seen/<map>/continuous_clips.json
  crossmap/<map>/support_seed_<0..4>.json
  crossmap/<map>/query_test.json
  crossmap/<map>/continuous_clips.json
aggregate/
  seen_train.json
  seen_validation.json
  seen_discrete_test.json
  crossmap_query_test.json
  crossmap_support_seed_<0..4>.json
benchmark_manifest.json
build_report.json
selected_images.sha256
checksums.sha256
```

The output root intentionally contains two artifact classes. The formal
runtime bundle for training, inference, and metric evaluation consists of
`benchmark_manifest.json`, `build_report.json`, `checksums.sha256`,
`selected_images.sha256`, `aggregate/`, `calibration/`, and `splits/`.
Within `calibration/`, the released `z_calibration.json` and
`z_extrema_rows.jsonl` are retained release/verification metadata. They must
not be deleted merely because a particular loader does not open every file
directly. The source FPV images selected by the manifests and
`selected_images.sha256`, together with every manifest-referenced radar file,
are also part of the runtime bundle.

The audit/rebuild workspace is separate from that bundle. `audit/` contains
candidate lists, audit reports, neighboring-frame context images, extrema and
jump visualizations, review exports, and related intermediate files;
`audit_archive/` contains historical audit snapshots. These files are needed
only to re-run audit/review or to reconstruct and rebuild a benchmark release.
The `audit/` and `audit_archive/` trees, as well as
`calibration/*.template.yaml` approval templates, are build-only artifacts
intended to be excluded by `.gitignore`; they are not required on a training,
inference, or metric-evaluation server after the formal runtime bundle has
been verified. Excluding these files from a server migration does not make the
formal runtime files above disposable.

Discrete manifests preserve the original pose-row schema. Continuous files
contain a metadata object and a `clips` list; each clip records its map,
record, clip ID, and ordered original rows. The benchmark manifest records
config/source/audit/decision/calibration fingerprints, pool identities,
counts, seeds, frozen per-map Z ranges, and the accepted anomaly policy.
`selected_images.sha256` binds every unique
FPV image referenced by a formal manifest without hashing millions of unused
source images.

### Consumer integration gate

The v2 consumer gate is closed at the source level. Legacy configurations
without `benchmark_v2_manifest` still use `splits_20000_5000`; v2 is enabled
only by explicitly selecting the manifest and split. The implemented path:

- selects `seen_train`, `seen_validation`, `seen_discrete_test`,
  `seen_continuous`, `crossmap_support`, `crossmap_query_test`, or
  `crossmap_continuous` from the manifest;
- accepts `benchmark_v2_support_seed` and `benchmark_v2_shots_per_map` for
  deterministic nested support subsets;
- loads the manifest's full-corpus per-map Z ranges and explicit radar paths;
- preserves continuous clip ID, frame order, and frame index through inference;
- lets inference override `--benchmark_v2_split`,
  `--benchmark_v2_maps`, `--benchmark_v2_support_seed`, and
  `--benchmark_v2_shots_per_map`, as well as deterministic output/checkpoint/
  seed settings;
- restricts generation metric coverage to selected manifest rows and evaluates
  continuous metrics on exact manifest clips;
- writes strict localization summaries as `benchmark_csgo_v2_loc.json` with
  equal-map macro metrics and inference/checkpoint provenance; and
- writes inference/metric provenance including the selected split, support
  seed, shots, and checkpoint.

The implementation has been checked by source compilation, selection/contract
smoke tests, and artifact checksum verification. The localization empirical
gate is complete for the `exp33_loc`/`exp34_loc` 100/50/20/10-shot curves;
other v2 model families still follow the commands in `record.md`. The
experiment matrix and adaptation protocol are specified in
[`CSGO_BENCHMARK_V2_TASK_SETTING.md`](CSGO_BENCHMARK_V2_TASK_SETTING.md), and
current results are tracked in
[`csgo_benchmark_v2_experiments_results.md`](csgo_benchmark_v2_experiments_results.md).

### Metric aggregation

The formal main-table/appendix metric set and Chinese field-by-field
explanations of the result JSONs are maintained in
[`CSGO_BENCHMARK_METRICS_ZH.md`](CSGO_BENCHMARK_METRICS_ZH.md). In particular,
learned external-localizer metrics may remain in historical JSONs but are not
part of the current formal reporting set.

After strict per-map evaluation has written one v2 JSON result per map, use
`scripts/aggregate_csgo_benchmark_v2_metrics.py maps` for the single-run
equal-map macro. Its exact arguments are `--manifest --split --input_root
--kind --output`; the map list is taken from the manifest. For example:

```bash
python scripts/aggregate_csgo_benchmark_v2_metrics.py maps \
  --manifest data/csgo_benchmark_v2/benchmark_manifest.json \
  --split seen_discrete_test \
  --input_root outputs_eval/benchmark_v2/exp31/seen/discrete \
  --kind discrete \
  --output outputs_eval/benchmark_v2/exp31/seen/discrete/summary.json

python scripts/aggregate_csgo_benchmark_v2_metrics.py maps \
  --manifest data/csgo_benchmark_v2/benchmark_manifest.json \
  --split seen_continuous \
  --input_root outputs_eval/benchmark_v2/exp31/seen/continuous \
  --kind continuous \
  --output outputs_eval/benchmark_v2/exp31/seen/continuous/summary.json
```

`eval_csgo_loc.py` already emits the strict localization summary, so
localization does not use `maps --kind localization`. For generation, run the
`maps` subcommand once on the `seed_0` output to produce the equal-map macro.
Do not run the `seeds` subcommand for `exp33*` or `exp34*`: one support draw
cannot define support-selection variance or a confidence interval. The generic
`seeds` subcommand remains available for optional future multi-draw studies.

Keep inference `--seed 42` fixed while passing the selected support draw
separately as `--benchmark_v2_support_seed 0`. Continuous FVD uses 16-frame,
stride-16 windows inside each exact 64-frame manifest clip; a window never
crosses a clip boundary. The v2 metric scripts reject incomplete coverage or a
missing `inference_manifest.json` by default; the debug override flags are not
part of the reported protocol.

## 9. Reproducible Build Commands

Run from `/home/jiahao/task/UniLIP`. These commands are intentionally manual;
the repository does not build or modify the real benchmark as a side effect of
importing the script.

### Step 1: inspect the frozen configuration

```bash
python scripts/build_csgo_benchmark_v2.py --help
python -m unittest discover -s tests -p 'test_build_csgo_benchmark_v2.py' -v
sed -n '1,240p' csgo_configs/benchmark_v2.yaml
```

### Step 2: generate the read-only audit artifacts

```bash
python scripts/build_csgo_benchmark_v2.py audit \
  --config csgo_configs/benchmark_v2.yaml
```

If an earlier audit exists and replacement is intentional:

```bash
python scripts/build_csgo_benchmark_v2.py audit \
  --config csgo_configs/benchmark_v2.yaml \
  --overwrite
```

### Step 3: review coordinate candidates

```bash
python -m json.tool data/csgo_benchmark_v2/audit/audit_report.json | less
less data/csgo_benchmark_v2/audit/coordinate_candidates.jsonl
less data/csgo_benchmark_v2/audit/coordinate_candidates.csv
cp data/csgo_benchmark_v2/audit/anomaly_decisions.template.yaml \
  csgo_configs/benchmark_v2_anomaly_decisions.yaml
```

Edit the copied decision file only after inspecting candidate images and their
neighboring captured frames. Set `status: approved`, fill
reviewer/date/references, select `keep` or `exclude`, and list all exceptions.
Keep the generated audit fingerprint, candidate-file hash, and line count
unchanged; the builder verifies all three.

For a provisional extrema review before anomaly approval, export the current
decision state without mutating source data:

```bash
python scripts/export_csgo_benchmark_v2_review.py \
  --config csgo_configs/benchmark_v2.yaml \
  --decisions csgo_configs/benchmark_v2_anomaly_decisions.yaml \
  --overwrite
```

This writes compact candidate/extrema CSV files under
`data/csgo_benchmark_v2/audit/review_exports`. They are review aids, not the
approved calibration consumed by `build`.

### Step 4: generate and review full-corpus Z calibration

```bash
python scripts/build_csgo_benchmark_v2.py calibrate \
  --config csgo_configs/benchmark_v2.yaml \
  --decisions csgo_configs/benchmark_v2_anomaly_decisions.yaml

less data/csgo_benchmark_v2/calibration/z_extrema_rows.jsonl
cp data/csgo_benchmark_v2/calibration/z_calibration_approval.template.yaml \
  csgo_configs/benchmark_v2_z_calibration_approval.yaml
```

Inspect every tied minimum/maximum frame and its captured neighbors. In the
copied calibration approval, fill reviewer/date/references and set
`review.status: approved`. Do not edit any generated fingerprint or Z range.

### Step 5: construct the formal manifests

```bash
python scripts/build_csgo_benchmark_v2.py build \
  --config csgo_configs/benchmark_v2.yaml \
  --decisions csgo_configs/benchmark_v2_anomaly_decisions.yaml \
  --calibration-approval csgo_configs/benchmark_v2_z_calibration_approval.yaml
```

### Step 6: independently validate the artifacts

```bash
python scripts/build_csgo_benchmark_v2.py validate \
  --config csgo_configs/benchmark_v2.yaml \
  --decisions csgo_configs/benchmark_v2_anomaly_decisions.yaml \
  --calibration-approval csgo_configs/benchmark_v2_z_calibration_approval.yaml
```

### Step 7: archive provenance

```bash
sha256sum csgo_configs/benchmark_v2.yaml \
  csgo_configs/benchmark_v2_anomaly_decisions.yaml \
  csgo_configs/benchmark_v2_z_calibration_approval.yaml \
  data/csgo_benchmark_v2/calibration/z_calibration.json \
  data/csgo_benchmark_v2/calibration/z_extrema_rows.jsonl \
  data/csgo_benchmark_v2/benchmark_manifest.json \
  data/csgo_benchmark_v2/build_report.json \
  data/csgo_benchmark_v2/selected_images.sha256
```

Do not use `--overwrite` for a released benchmark version. Change the protocol
version and output root instead.

## 10. Acceptance Criteria

A v2 build is publishable only when all of the following are true:

- all 14 intended maps and no `_v0` alias are selected;
- audit source hashes match the build source hashes;
- the anomaly decision is approved and tied to that audit fingerprint;
- every requested frame/clip quota is exact;
- Seen record pools are pairwise disjoint;
- CrossMap support, query, and continuous record pools are pairwise disjoint;
- continuous clips have exact length, legal frame gaps, no reused frames, and
  retained clip boundaries;
- aggregate manifests exactly equal the union of per-map manifests;
- every selected FPV image path/content matches `selected_images.sha256`;
- generated-file checksums pass independent validation;
- the approved full-corpus per-map Z calibration and every tied extrema row are
  fingerprinted, and the model loader performs no implicit clipping or
  split-derived fallback;
- map file versions, capture snapshot, config, decisions, and evaluation
  checkpoint provenance are archived.

## 11. Known Scope Boundaries

- The builder controls split leakage; it does not establish that every
  `file_num` is an independently captured physical trajectory.
- The provisional coordinate thresholds are not authoritative map bounds.
- The protocol does not solve external-localizer leakage. Any learned metric
  model must obey the same Seen-10/CrossMap-4 boundary or be disclosed as
  externally supervised.
- Zero-shot and 100-shot claims require that base-model parameter optimization
  has not consumed CrossMap captured images or pose rows. Evaluation does use
  the published per-map Z extrema derived from the complete CrossMap corpus,
  so the precise claim is zero-shot model adaptation under frozen benchmark
  map-level calibration, not target-metadata-free domain generalization.
- A 100-shot result alone is a low-shot adaptation result, not evidence of
  general few-shot scaling. The implemented localization protocol adds nested
  10/20/50-shot points under the same fixed query protocol; 1/5-shot behavior
  remains outside the current experiment matrix.
