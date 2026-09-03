#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
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

log_dir="out/zh-training-dynamics/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tbatch_order\tnegative_set\tlr_schedule\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local batch_order="$3"
  local negative_set="$4"
  local lr_schedule="$5"
  local run_name="zh_en_exact_${label}_q32_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$batch_order" "$negative_set" "$lr_schedule" >> "$manifest"
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
    --lr 1e-6 \
    --lr_schedule "$lr_schedule" \
    --batch_order "$batch_order" \
    --negative_set "$negative_set" \
    --eval_every_epochs 1 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  child_pids+=("$!")
  echo "started pid=$! gpu=$gpu run=$run_name"
}

child_pids=()
run_one "${gpus[0]}" fixed_order fixed queue-only constant
run_one "${gpus[1]}" paper_count reshuffle paper-count constant
run_one "${gpus[2]}" author_periodic reshuffle queue-only author-periodic
run_one "${gpus[3]}" fixed_paper_count fixed paper-count constant

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "training_dynamics_complete status=$status manifest=$manifest"
exit "$status"
