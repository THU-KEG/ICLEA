#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
use_fnmask="${USE_FNMASK:-0}"
config_set="${CONFIG_SET:-all}"
IFS=',' read -r -a gpus <<< "$gpu_csv"

case "$config_set" in
  all) expected_gpus=4 ;;
  high-lr|remaining) expected_gpus=2 ;;
  *)
    echo "CONFIG_SET must be all, high-lr, or remaining" >&2
    exit 2
    ;;
esac
if [[ "${#gpus[@]}" -ne "$expected_gpus" ]]; then
  echo "GPU_IDS must contain $expected_gpus distinct physical GPUs from 0-3 for CONFIG_SET=$config_set" >&2
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

log_dir="out/zh-reverse-refinement/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest_${config_set}.tsv"
printf 'run_name\tphysical_gpu\tconfig_set\tlr\treverse_icl_weight\texclude_false_negatives\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local lr="$3"
  local reverse_weight="$4"
  local force_fnmask="$5"
  local effective_fnmask="$use_fnmask"
  if [[ "$force_fnmask" == 1 ]]; then
    effective_fnmask=1
  fi
  local mask_args=()
  if [[ "$effective_fnmask" == 1 ]]; then
    mask_args+=(--exclude_false_negatives)
  fi
  local run_name="zh_en_exact_${label}_paper_count_q32_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$config_set" "$lr" "$reverse_weight" "$effective_fnmask" >> "$manifest"
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
    --dropout 0.0 \
    --reverse_icl_weight "$reverse_weight" \
    --momentum_init copy-online \
    --pair_mining l2 \
    --eval_every_epochs 1 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  child_pids+=("$!")
  echo "started pid=$! gpu=$gpu run=$run_name"
}

child_pids=()
case "$config_set" in
  all)
    run_one "${gpus[0]}" fnmask_lr2e6_reverse05 2e-6 0.5 1
    run_one "${gpus[1]}" lr2e6_reverse075 2e-6 0.75 0
    run_one "${gpus[2]}" lr25e6_reverse05 2.5e-6 0.5 0
    run_one "${gpus[3]}" lr3e6_reverse05 3e-6 0.5 0
    ;;
  high-lr)
    run_one "${gpus[0]}" lr25e6_reverse05 2.5e-6 0.5 0
    run_one "${gpus[1]}" lr3e6_reverse05 3e-6 0.5 0
    ;;
  remaining)
    run_one "${gpus[0]}" fnmask_lr2e6_reverse05 2e-6 0.5 1
    run_one "${gpus[1]}" lr2e6_reverse075 2e-6 0.75 0
    ;;
esac

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "reverse_refinement_complete status=$status manifest=$manifest"
exit "$status"
