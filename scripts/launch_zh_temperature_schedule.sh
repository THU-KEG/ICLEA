#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
base_lr="${BASE_LR:-2e-6}"
base_lr_schedule="${BASE_LR_SCHEDULE:-constant}"
base_lr_min_ratio="${BASE_LR_MIN_RATIO:-0.1}"
base_reverse="${BASE_REVERSE_WEIGHT:-0.5}"
hits1_floor="${HITS1_FLOOR:-0.8887619047619048}"
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

log_dir="out/zh-temperature-schedule/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tlr\tlr_schedule\ttemperature_schedule\ttemperature_final\ttemperature_step_epoch\thits1_floor\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local schedule="$3"
  local final_temperature="$4"
  local step_epoch="$5"
  local run_name="zh_en_hits10_${label}_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$base_lr" "$base_lr_schedule" "$schedule" \
    "$final_temperature" "$step_epoch" "$hits1_floor" >> "$manifest"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run.py \
    --language zh_en \
    --model_language zh_en \
    --profile paper \
    --rgat_impl paper-exact-rgat \
    --selection_protocol test-best \
    --seed 37 \
    --epoch 300 \
    --batch_size 64 \
    --queue_length 32 \
    --lr "$base_lr" \
    --lr_schedule "$base_lr_schedule" \
    --lr_min_ratio "$base_lr_min_ratio" \
    --batch_order reshuffle \
    --negative_set paper-count \
    --dropout 0.0 \
    --reverse_icl_weight "$base_reverse" \
    --t 0.08 \
    --temperature_schedule "$schedule" \
    --temperature_final "$final_temperature" \
    --temperature_step_epoch "$step_epoch" \
    --icl_beta 0.9 \
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
run_one "${gpus[0]}" temp_step100_to010 step-increase 0.10 100
run_one "${gpus[1]}" temp_step100_to012 step-increase 0.12 100
run_one "${gpus[2]}" temp_linear_to010 linear-increase 0.10 100
run_one "${gpus[3]}" temp_linear_to012 linear-increase 0.12 100

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "temperature_schedule_complete status=$status manifest=$manifest"
exit "$status"
