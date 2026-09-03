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
use_fnmask="${USE_FNMASK:-1}"
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
if [[ "$use_fnmask" != 0 && "$use_fnmask" != 1 ]]; then
  echo "USE_FNMASK must be 0 or 1" >&2
  exit 2
fi
mask_args=()
if [[ "$use_fnmask" == 1 ]]; then
  mask_args+=(--exclude_false_negatives)
fi

log_dir="out/zh-temperature-beta/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tlr\tlr_schedule\tlr_min_ratio\treverse_icl_weight\ttemperature\ticl_beta\ticl_source_inbatch\texclude_false_negatives\thits1_floor\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local temperature="$3"
  local beta="$4"
  local source_inbatch="$5"
  local run_name="zh_en_exact_${label}_paper_count_q32_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$base_lr" "$base_lr_schedule" "$base_lr_min_ratio" \
    "$base_reverse" "$temperature" "$beta" "$source_inbatch" "$use_fnmask" "$hits1_floor" \
    >> "$manifest"
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
    "${mask_args[@]}" \
    --dropout 0.0 \
    --reverse_icl_weight "$base_reverse" \
    --t "$temperature" \
    --icl_beta "$beta" \
    --icl_source_inbatch "$source_inbatch" \
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
run_one "${gpus[0]}" tau008_beta09_legacy_fnmask_control 0.08 0.9 pseudo-target
run_one "${gpus[1]}" tau008_beta09_paper_source 0.08 0.9 source
run_one "${gpus[2]}" tau010_beta09_paper_source 0.10 0.9 source
run_one "${gpus[3]}" tau008_beta095_paper_source 0.08 0.95 source

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "temperature_beta_complete status=$status manifest=$manifest"
exit "$status"
