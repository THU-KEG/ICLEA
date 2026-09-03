#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
dropout="${BASE_DROPOUT:-0.0}"
use_fnmask="${USE_FNMASK:-0}"
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

log_dir="out/zh-lr-refinement/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tlr\tlr_schedule\tlr_min_ratio\tdropout\texclude_false_negatives\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local lr="$3"
  local schedule="$4"
  local min_ratio="$5"
  local run_name="zh_en_exact_${label}_paperloss_csls1_q32_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$lr" "$schedule" "$min_ratio" "$dropout" "$use_fnmask" \
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
    --lr "$lr" \
    --lr_schedule "$schedule" \
    --lr_min_ratio "$min_ratio" \
    --batch_order reshuffle \
    --negative_set paper-count \
    "${mask_args[@]}" \
    --dropout "$dropout" \
    --reverse_icl_weight 0.0 \
    --momentum_init copy-online \
    --pair_mining csls \
    --pair_csls_k 1 \
    --pair_candidate_pool 100 \
    --eval_every_epochs 1 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  child_pids+=("$!")
  echo "started pid=$! gpu=$gpu run=$run_name"
}

child_pids=()
run_one "${gpus[0]}" lr15e6_constant 1.5e-6 constant 0.1
run_one "${gpus[1]}" lr25e6_constant 2.5e-6 constant 0.1
run_one "${gpus[2]}" lr30e6_constant 3e-6 constant 0.1
run_one "${gpus[3]}" lr30e6_cosine 3e-6 cosine 0.1

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "lr_refinement_complete status=$status manifest=$manifest"
exit "$status"
