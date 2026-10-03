export PATH=/home/jiahao/miniconda3/envs/UniLIP/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

set -euo pipefail
V2_MANIFEST=data/csgo_benchmark_v2/benchmark_manifest.json
V2_ASSET_MANIFEST=data/csgo_benchmark_v2/minimal_dataset_report.json
V2_GT_ROOT=data/csgo_benchmark_v2/images
SEEN_MAPS=(cs_agency cs_italy de_ancient de_anubis de_dust2 de_inferno de_mirage de_nuke de_overpass de_train)
CROSS_MAPS=(cs_office de_golden de_palacio de_vertigo)
EXTERNAL_LOC_ROOT=csgosquare
EXTERNAL_LOC_CONFIG=configs_reg_newdata/exp5_2.yaml
EXTERNAL_LOC_CKPT=checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth
EXPERIMENT=exp32_1

run_discrete() {
  local input_root="$1"
  local split="$2"
  shift 2
  local map_name
  for map_name in "$@"; do
    CUDA_VISIBLE_DEVICES=0 python benchmark_csgo_v1.py     --gt "${V2_GT_ROOT}/${map_name}"     --pred "${input_root}/gen_imgs/${map_name}"     --batch_size 1     --device cuda     --paired_size 448     --data_dir data/preprocessed_data     --map_name "$map_name"     --benchmark_v2_manifest "$V2_MANIFEST"     --benchmark_v2_asset_manifest "$V2_ASSET_MANIFEST"     --benchmark_v2_split "$split"     --external_loc_repo_root "$EXTERNAL_LOC_ROOT"     --external_loc_config_path "$EXTERNAL_LOC_CONFIG"     --external_loc_checkpoint_path "$EXTERNAL_LOC_CKPT"     --metric_profile benchmark_v2_core
  done
}

run_continuous() {
  local input_root="$1"
  local split="$2"
  shift 2
  local map_name
  for map_name in "$@"; do
    CUDA_VISIBLE_DEVICES=0 python benchmark_csgo_v1_conti.py     --gt "${V2_GT_ROOT}/${map_name}"     --pred "${input_root}/gen_imgs/${map_name}"     --batch_size 1     --device cuda     --paired_size 448     --data_dir data/preprocessed_data     --map_name "$map_name"     --frame_diff_threshold 2     --min_track_len 4     --clip_length 16     --clip_stride 16     --fvd_size 224     --benchmark_v2_manifest "$V2_MANIFEST"     --benchmark_v2_asset_manifest "$V2_ASSET_MANIFEST"     --benchmark_v2_split "$split"     --external_loc_repo_root "$EXTERNAL_LOC_ROOT"     --external_loc_config_path "$EXTERNAL_LOC_CONFIG"     --external_loc_checkpoint_path "$EXTERNAL_LOC_CKPT"     --metric_profile benchmark_v2_core
  done
}

aggregate() {
  local input_root="$1"
  local split="$2"
  local kind="$3"
  python scripts/aggregate_csgo_benchmark_v2_metrics.py maps   --manifest "$V2_MANIFEST"   --split "$split"   --input_root "$input_root"   --kind "$kind"   --output "${input_root}/summary.json"
}

run_discrete "outputs_eval/benchmark_v2/${EXPERIMENT}/seen/discrete" seen_discrete_test "${SEEN_MAPS[@]}"
aggregate "outputs_eval/benchmark_v2/${EXPERIMENT}/seen/discrete" seen_discrete_test discrete
run_continuous "outputs_eval/benchmark_v2/${EXPERIMENT}/seen/continuous" seen_continuous "${SEEN_MAPS[@]}"
aggregate "outputs_eval/benchmark_v2/${EXPERIMENT}/seen/continuous" seen_continuous continuous
