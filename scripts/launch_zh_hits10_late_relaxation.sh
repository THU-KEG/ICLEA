#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
hits1_floor="${HITS1_FLOOR:-0.8887619047619048}"
base_queue="${BASE_QUEUE_LENGTH:-32}"
base_lr="${BASE_LR:-2e-6}"
base_lr_schedule="${BASE_LR_SCHEDULE:-constant}"
base_lr_min_ratio="${BASE_LR_MIN_RATIO:-0.1}"
base_lr_step_size="${BASE_LR_STEP_SIZE:-20}"
base_lr_decay="${BASE_LR_DECAY:-0.9}"
base_reverse="${BASE_REVERSE_WEIGHT:-0.5}"
base_temperature="${BASE_TEMPERATURE:-0.08}"
final_temperature="${FINAL_TEMPERATURE:-0.10}"
base_beta="${BASE_ICL_BETA:-0.9}"
base_source_inbatch="${BASE_ICL_SOURCE_INBATCH:-source}"
use_fnmask="${USE_FNMASK:-1}"
switch_epoch="${SWITCH_EPOCH:-100}"
IFS=',' read -r -a gpus <<< "$gpu_csv"

if [[ "${#gpus[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four distinct physical GPUs from 0-3" >&2
  exit 2
fi
declare -A seen=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-3]$ ]] || [[ -n "${seen[$gpu]:-}" ]]; then
    echo "GPU_IDS must be a permutation of distinct physical GPUs 0-3" >&2
    exit 2
  fi
  seen[$gpu]=1
done
if [[ "$use_fnmask" != 0 && "$use_fnmask" != 1 ]]; then
  echo "USE_FNMASK must be 0 or 1" >&2
  exit 2
fi
if ! "$python_bin" -c 'import sys; base,final=map(float,sys.argv[1:]); assert final >= base > 0' "$base_temperature" "$final_temperature"; then
  echo "FINAL_TEMPERATURE must be no smaller than BASE_TEMPERATURE" >&2
  exit 2
fi
mask_args=()
if [[ "$use_fnmask" == 1 ]]; then
  mask_args+=(--exclude_false_negatives)
fi

log_dir="out/zh-hits10-late-relaxation/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tqueue_length\tlr\tlr_schedule\ttemperature_schedule\ttemperature_initial\ttemperature_final\treverse_schedule\treverse_initial\treverse_final\tswitch_epoch\ticl_beta\ticl_source_inbatch\texclude_false_negatives\thits1_floor\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local temperature_schedule="$3"
  local temperature_final="$4"
  local reverse_schedule="$5"
  local reverse_final="$6"
  local run_name="zh_en_hits10_relax_${label}_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$base_queue" "$base_lr" "$base_lr_schedule" \
    "$temperature_schedule" "$base_temperature" "$temperature_final" \
    "$reverse_schedule" "$base_reverse" "$reverse_final" "$switch_epoch" \
    "$base_beta" "$base_source_inbatch" "$use_fnmask" "$hits1_floor" >> "$manifest"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run.py \
    --language zh_en \
    --model_language zh_en \
    --profile paper \
    --rgat_impl paper-exact-rgat \
    --selection_protocol test-best \
    --seed 37 \
    --epoch 300 \
    --batch_size 64 \
    --queue_length "$base_queue" \
    --lr "$base_lr" \
    --lr_schedule "$base_lr_schedule" \
    --lr_min_ratio "$base_lr_min_ratio" \
    --lr_step_size "$base_lr_step_size" \
    --lr_decay "$base_lr_decay" \
    --batch_order reshuffle \
    --negative_set paper-count \
    "${mask_args[@]}" \
    --dropout 0.0 \
    --reverse_icl_weight "$base_reverse" \
    --reverse_icl_schedule "$reverse_schedule" \
    --reverse_icl_final_weight "$reverse_final" \
    --reverse_icl_step_epoch "$switch_epoch" \
    --t "$base_temperature" \
    --temperature_schedule "$temperature_schedule" \
    --temperature_final "$temperature_final" \
    --temperature_step_epoch "$switch_epoch" \
    --icl_beta "$base_beta" \
    --icl_source_inbatch "$base_source_inbatch" \
    --momentum_init copy-online \
    --pair_mining l2 \
    --eval_every_epochs 1 \
    --joint_hits1_floor "$hits1_floor" \
    --snapshot_epochs 75,100,150,200,250,299 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  child_pids+=("$!")
  echo "started pid=$! gpu=$gpu run=$run_name"
}

child_pids=()
run_one "${gpus[0]}" temp_step100 step-increase "$final_temperature" constant "$base_reverse"
run_one "${gpus[1]}" temp_linear100 linear-increase "$final_temperature" constant "$base_reverse"
run_one "${gpus[2]}" reverse_step100_to0 constant "$base_temperature" step-decay 0.0
run_one "${gpus[3]}" reverse_linear100_to0 constant "$base_temperature" linear-decay 0.0

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "hits10_late_relaxation_complete status=$status manifest=$manifest"
exit "$status"
