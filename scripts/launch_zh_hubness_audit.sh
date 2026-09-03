#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
base_dropout="${BASE_DROPOUT:-0.0}"
base_reverse="${BASE_REVERSE_WEIGHT:-0.0}"
base_lr="${BASE_LR:-1e-6}"
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

log_dir="out/zh-hubness/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tmomentum_init\tpair_mining\tpair_csls_k\tdropout\treverse_icl_weight\tlr\texclude_false_negatives\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local momentum_init="$3"
  local pair_mining="$4"
  local lr="$5"
  local csls_k="$6"
  local reverse_weight="$7"
  local run_name="zh_en_exact_${label}_paper_count_q32_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$momentum_init" "$pair_mining" "$csls_k" "$base_dropout" \
    "$reverse_weight" "$lr" "$use_fnmask" >> "$manifest"
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
    "${mask_args[@]}" \
    --dropout "$base_dropout" \
    --reverse_icl_weight "$reverse_weight" \
    --momentum_init "$momentum_init" \
    --pair_mining "$pair_mining" \
    --pair_csls_k "$csls_k" \
    --pair_candidate_pool 100 \
    --eval_every_epochs 1 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  child_pids+=("$!")
  echo "started pid=$! gpu=$gpu run=$run_name"
}

child_pids=()
run_one "${gpus[0]}" independent_l2_paperloss "independent" "l2" "$base_lr" 1 0.0
run_one "${gpus[1]}" copy_csls_k1_paperloss "copy-online" "csls" "$base_lr" 1 0.0
run_one "${gpus[2]}" copy_csls_k3_paperloss "copy-online" "csls" "$base_lr" 3 0.0
run_one "${gpus[3]}" independent_csls_k1_base "independent" "csls" "$base_lr" 1 "$base_reverse"

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "hubness_audit_complete status=$status manifest=$manifest"
exit "$status"
