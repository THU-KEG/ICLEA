#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1}"
IFS=',' read -r -a gpus <<< "$gpu_csv"

if [[ "${#gpus[@]}" -ne 2 ]]; then
  echo "GPU_IDS must contain exactly two distinct physical GPUs from 0-3" >&2
  exit 2
fi
if [[ "${gpus[0]}" == "${gpus[1]}" ]]; then
  echo "GPU_IDS entries must be distinct" >&2
  exit 2
fi
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-3]$ ]]; then
    echo "GPU $gpu is outside the allowed physical range 0-3" >&2
    exit 2
  fi
done

log_dir="out/zh-rgat-control-logs/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\trgat_impl\tphysical_gpu\tseed\tepochs\tqueue_length\tlr_schedule\tprofile\n' > "$manifest"

run_one() {
  local gpu="$1"
  local rgat_impl="$2"
  local safe_impl="${rgat_impl//-/_}"
  local run_name="zh_en_rgat_control_${safe_impl}_paper_q32_constlr_seed37_${control_tag}"
  printf '%s\t%s\t%s\t37\t300\t32\tconstant\tpaper\n' \
    "$run_name" "$rgat_impl" "$gpu" >> "$manifest"
  echo "start run=${run_name} rgat_impl=${rgat_impl} physical_gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run.py \
    --language zh_en \
    --model_language zh_en \
    --profile paper \
    --rgat_impl "$rgat_impl" \
    --selection_protocol test-best \
    --seed 37 \
    --epoch 300 \
    --batch_size 64 \
    --queue_length 32 \
    --lr 1e-6 \
    --lr_schedule constant \
    --eval_every_epochs 1 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  local child_pid="$!"
  echo "pid=${child_pid} run=${run_name} physical_gpu=${gpu}"
  child_pids+=("$child_pid")
}

child_pids=()
run_one "${gpus[0]}" author-code
run_one "${gpus[1]}" paper-exact-rgat

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=${child_pid}"
    status=1
  fi
done
echo "zh_rgat_control_complete status=${status} manifest=${manifest}"
exit "$status"
