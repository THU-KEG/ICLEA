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

log_dir="out/zh-false-negative/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tdropout\treverse_icl_weight\tlr\texclude_false_negatives\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local dropout="$3"
  local reverse_weight="$4"
  local lr="$5"
  local run_name="zh_en_exact_${label}_paper_count_q32_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\ttrue\n' \
    "$run_name" "$gpu" "$dropout" "$reverse_weight" "$lr" >> "$manifest"
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
    --lr "$lr" \
    --lr_schedule constant \
    --batch_order reshuffle \
    --negative_set paper-count \
    --exclude_false_negatives \
    --dropout "$dropout" \
    --reverse_icl_weight "$reverse_weight" \
    --eval_every_epochs 1 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  child_pids+=("$!")
  echo "started pid=$! gpu=$gpu run=$run_name"
}

child_pids=()
run_one "${gpus[0]}" fnmask_dropout0 0.0 0.0 1e-6
run_one "${gpus[1]}" fnmask_reverse05 0.3 0.5 1e-6
run_one "${gpus[2]}" fnmask_dropout0_reverse05 0.0 0.5 1e-6
run_one "${gpus[3]}" fnmask_dropout0_reverse05_lr2e6 0.0 0.5 2e-6

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "false_negative_complete status=$status manifest=$manifest"
exit "$status"
