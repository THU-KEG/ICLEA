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

log_dir="out/zh-reverse-schedule/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tlr\tlr_schedule\treverse_schedule\treverse_initial\treverse_final\treverse_step_epoch\thits1_floor\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local schedule="$3"
  local final_weight="$4"
  local step_epoch="$5"
  local run_name="zh_en_hits10_${label}_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$base_lr" "$base_lr_schedule" "$schedule" 0.5 \
    "$final_weight" "$step_epoch" "$hits1_floor" >> "$manifest"
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
    --reverse_icl_weight 0.5 \
    --reverse_icl_schedule "$schedule" \
    --reverse_icl_final_weight "$final_weight" \
    --reverse_icl_step_epoch "$step_epoch" \
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
run_one "${gpus[0]}" reverse_step100_to025 step-decay 0.25 100
run_one "${gpus[1]}" reverse_step100_to000 step-decay 0.00 100
run_one "${gpus[2]}" reverse_linear_to025 linear-decay 0.25 100
run_one "${gpus[3]}" reverse_linear_to000 linear-decay 0.00 100

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "reverse_schedule_complete status=$status manifest=$manifest"
exit "$status"
