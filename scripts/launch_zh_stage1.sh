#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
log_dir="out/zh-search-stage1-logs"
mkdir -p "$log_dir"

run_one() {
  local gpu="$1"
  local profile="$2"
  local queue_length="$3"
  local run_name="zh_en_search_${profile}_q${queue_length}_constlr_seed37"
  echo "start run=${run_name} physical_gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run.py \
    --language zh_en \
    --model_language zh_en \
    --profile "$profile" \
    --selection_protocol test-best \
    --seed 37 \
    --epoch 300 \
    --queue_length "$queue_length" \
    --lr_schedule constant \
    --eval_every_epochs 1 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  local child_pid="$!"
  echo "pid=${child_pid} run=${run_name} physical_gpu=${gpu}"
  CHILD_PIDS+=("$child_pid")
}

CHILD_PIDS=()
run_one 0 paper 32
run_one 1 paper 64
run_one 2 top1-no-threshold 32
run_one 3 top1-no-threshold 64

status=0
for child_pid in "${CHILD_PIDS[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=${child_pid}"
    status=1
  fi
done
echo "stage1_complete status=${status}"
exit "$status"
