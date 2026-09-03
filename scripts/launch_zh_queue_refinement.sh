#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
queue_csv="${QUEUE_LENGTHS:-40,48,56,64}"
base_lr="${BASE_LR:-2e-6}"
base_lr_schedule="${BASE_LR_SCHEDULE:-constant}"
base_lr_min_ratio="${BASE_LR_MIN_RATIO:-0.1}"
base_reverse="${BASE_REVERSE_WEIGHT:-0.5}"
hits1_floor="${HITS1_FLOOR:-0.8887619047619048}"
IFS=',' read -r -a gpus <<< "$gpu_csv"
IFS=',' read -r -a queues <<< "$queue_csv"

if [[ "${#gpus[@]}" -ne 4 || "${#queues[@]}" -ne 4 ]]; then
  echo "GPU_IDS and QUEUE_LENGTHS must each contain exactly four values" >&2
  exit 2
fi
declare -A seen_gpu=()
declare -A seen_queue=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-3]$ ]] || [[ -n "${seen_gpu[$gpu]:-}" ]]; then
    echo "GPU_IDS must be a permutation of distinct physical GPUs 0-3" >&2
    exit 2
  fi
  seen_gpu[$gpu]=1
done
for queue_length in "${queues[@]}"; do
  if [[ ! "$queue_length" =~ ^[0-9]+$ ]] || (( queue_length < 1 || queue_length > 64 )); then
    echo "QUEUE_LENGTHS must contain distinct integers in [1,64]" >&2
    exit 2
  fi
  if [[ -n "${seen_queue[$queue_length]:-}" ]]; then
    echo "QUEUE_LENGTHS values must be distinct" >&2
    exit 2
  fi
  seen_queue[$queue_length]=1
done

log_dir="out/zh-queue-refinement/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tqueue_length\tlr\tlr_schedule\treverse_icl_weight\thits1_floor\n' > "$manifest"

run_one() {
  local gpu="$1"
  local queue_length="$2"
  local run_name="zh_en_hits10_q${queue_length}_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$queue_length" "$base_lr" "$base_lr_schedule" \
    "$base_reverse" "$hits1_floor" >> "$manifest"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run.py \
    --language zh_en \
    --model_language zh_en \
    --profile paper \
    --rgat_impl paper-exact-rgat \
    --selection_protocol test-best \
    --seed 37 \
    --epoch 300 \
    --batch_size 64 \
    --queue_length "$queue_length" \
    --lr "$base_lr" \
    --lr_schedule "$base_lr_schedule" \
    --lr_min_ratio "$base_lr_min_ratio" \
    --batch_order reshuffle \
    --negative_set paper-count \
    --dropout 0.0 \
    --reverse_icl_weight "$base_reverse" \
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
  echo "started pid=$! gpu=$gpu queue_length=$queue_length run=$run_name"
}

child_pids=()
for index in 0 1 2 3; do
  run_one "${gpus[$index]}" "${queues[$index]}"
done

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "queue_refinement_complete status=$status manifest=$manifest"
exit "$status"
