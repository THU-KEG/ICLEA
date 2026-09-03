#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
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

log_dir="out/zh-lr-single-step/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tinitial_lr\tstep_epoch\tdecay\tfinal_lr\tintegrated_lr_units\thits1_floor\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local initial_lr="$3"
  local step_epoch="$4"
  local decay="$5"
  local final_lr integrated
  final_lr="$($python_bin -c "print(float('$initial_lr') * float('$decay'))")"
  integrated="$($python_bin -c "print(float('$initial_lr') * int('$step_epoch') + float('$final_lr') * (300-int('$step_epoch')))")"
  local run_name="zh_en_hits10_${label}_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$initial_lr" "$step_epoch" "$decay" "$final_lr" \
    "$integrated" "$hits1_floor" >> "$manifest"
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
    --lr "$initial_lr" \
    --lr_schedule single-step \
    --lr_step_size "$step_epoch" \
    --lr_decay "$decay" \
    --batch_order reshuffle \
    --negative_set paper-count \
    --dropout 0.0 \
    --reverse_icl_weight 0.5 \
    --reverse_icl_schedule constant \
    --t 0.08 \
    --temperature_schedule constant \
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
run_one "${gpus[0]}" lr3e6_step100_to15e6 3e-6 100 0.5
run_one "${gpus[1]}" lr3e6_step75_to1667e6 3e-6 75 0.5555555555555556
run_one "${gpus[2]}" lr25e6_step100_to175e6 2.5e-6 100 0.7
run_one "${gpus[3]}" lr25e6_step75_to1833e6 2.5e-6 75 0.7333333333333333

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "lr_single_step_complete status=$status manifest=$manifest"
exit "$status"
