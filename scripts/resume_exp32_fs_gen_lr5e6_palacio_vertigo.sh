#!/usr/bin/env bash
# Resume this queue only; completed models and inference results are reused.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}"
GPU="${GPU:-1}"
FAMILY=exp32_fs_gen_lr5e6

mkdir -p fst_outs
exec 9>fst_outs/exp32_fs_gen_lr5e6_palacio_vertigo.resume.lock
if ! flock -n 9; then
  printf 'Another copy of this recovery queue is already running.\n' >&2
  exit 1
fi
trap 'resume_rc=$?; printf "Failed at line %s (exit %s). Existing results are retained.\n" "$LINENO" "$resume_rc" >&2; exit "$resume_rc"' ERR

COMMON=(
  --experiment "$FAMILY" --test-gen-batch-size 32
  --cuda-device "$GPU" --seed 42 --support-seed 0
  --report-to none --asset-mode minimal --split query --master-port 0
)

run_stage() {
  local stage=$1 map_name=$2 shots=$3 task=$4
  printf '\n%s %s %s shot_%s task=%s\n' "$(date -Is)" "$stage" "$map_name" "$shots" "$task"
  "$PYTHON_BIN" -u scripts/run_exp32_gen_unseen4.py "$stage" "${COMMON[@]}" \
    --map "$map_name" --shots "$shots" --task "$task" --execute
}

for MAP in de_palacio de_vertigo; do
  if [[ "$MAP" == de_palacio ]]; then
    SHOTS=(50 20 10)
  else
    SHOTS=(100 50 20 10)
  fi
  for SHOT in "${SHOTS[@]}"; do
    MODEL_DIR="outputs/csgo_1b/${FAMILY}_${MAP}/shot_${SHOT}/seed_0"
    TRAINING_STATUS=$("$PYTHON_BIN" -c \
      'import sys; from pathlib import Path; from scripts.run_exp32_gen_unseen4 import adaptation_status; print(adaptation_status(Path(sys.argv[1]))["training_status"])' \
      "$MODEL_DIR")
    case "$TRAINING_STATUS" in
      complete)
        printf 'Reuse completed 400-step model: %s\n' "$MODEL_DIR"
        ;;
      missing)
        run_stage train "$MAP" "$SHOT" all
        ;;
      *)
        printf 'Training is %s, not a completed 400-step model: %s. Check its training log before restarting.\n' \
          "$TRAINING_STATUS" "$MODEL_DIR" >&2
        exit 1
        ;;
    esac

    for TASK in gen gen_continuous; do
      if [[ "$TASK" == gen ]]; then
        KIND=discrete
        METRIC="benchmark_csgo_v2_${MAP}.json"
      else
        KIND=continuous
        METRIC="benchmark_csgo_v2_conti_${MAP}.json"
      fi
      EVAL_DIR="outputs_eval/benchmark_v2/${FAMILY}_${MAP}/shot_${SHOT}/seed_0/infer_bs32/$KIND"
      if [[ ! -s "$EVAL_DIR/inference_manifest.json" ]]; then
        if [[ -e "$EVAL_DIR" ]]; then
          BACKUP="${EVAL_DIR}.partial.$(date +%Y%m%d_%H%M%S).$$"
          mv -- "$EVAL_DIR" "$BACKUP"
          printf 'Archived unfinished inference: %s\n' "$BACKUP"
        fi
        run_stage infer "$MAP" "$SHOT" "$TASK"
      else
        printf 'Reuse completed inference: %s\n' "$EVAL_DIR"
      fi
      if [[ ! -s "$EVAL_DIR/$METRIC" ]]; then
        run_stage metrics "$MAP" "$SHOT" "$TASK"
      else
        printf 'Reuse existing metric: %s/%s\n' "$EVAL_DIR" "$METRIC"
      fi
    done
  done
done
printf '\nThe lr5e6 palacio/vertigo training, inference and metrics queue is complete.\n'
