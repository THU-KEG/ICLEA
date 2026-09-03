#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
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

log_dir="out/zh-hits10/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tlr\tlr_schedule\tlr_min_ratio\tsnapshot_epochs\thits1_floor\n' > "$manifest"

run_one() {
  local gpu="$1"
  local label="$2"
  local lr="$3"
  local schedule="$4"
  local min_ratio="$5"
  local snapshots="$6"
  local run_name="zh_en_hits10_${label}_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$lr" "$schedule" "$min_ratio" "$snapshots" \
    "$hits1_floor" >> "$manifest"
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
    --dropout 0.0 \
    --reverse_icl_weight 0.5 \
    --t 0.08 \
    --icl_beta 0.9 \
    --momentum_init copy-online \
    --pair_mining l2 \
    --eval_every_epochs 1 \
    --joint_hits1_floor "$hits1_floor" \
    --snapshot_epochs "$snapshots" \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  child_pids+=("$!")
  echo "started pid=$! gpu=$gpu run=$run_name"
}

child_pids=()
run_one "${gpus[0]}" lr2e6_constant_control 2e-6 constant 0.1 95,150,200,250,299
run_one "${gpus[1]}" lr25e6_constant 2.5e-6 constant 0.1 75,100,150,200,250,299
run_one "${gpus[2]}" lr3e6_constant 3e-6 constant 0.1 75,100,150,200,250,299
run_one "${gpus[3]}" lr3e6_cosine025 3e-6 cosine 0.25 75,100,150,200,250,299

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "hits10_wave1_complete status=$status manifest=$manifest"
exit "$status"
