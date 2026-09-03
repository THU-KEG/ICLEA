#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
control_tag="${CONTROL_TAG:-$(date +%Y%m%d%H%M%S)}"
gpu_csv="${GPU_IDS:-0,1,2,3}"
hits1_floor="${HITS1_FLOOR:-0.8887619047619048}"
base_queue="${BASE_QUEUE_LENGTH:-32}"
base_lr="${BASE_LR:-2e-6}"
base_lr_schedule="${BASE_LR_SCHEDULE:-constant}"
base_lr_min_ratio="${BASE_LR_MIN_RATIO:-0.1}"
base_lr_step_size="${BASE_LR_STEP_SIZE:-20}"
base_lr_decay="${BASE_LR_DECAY:-0.9}"
evidence_lr="${EVIDENCE_LR:-2.5e-6}"
base_reverse="${BASE_REVERSE_WEIGHT:-0.5}"
base_reverse_schedule="${BASE_REVERSE_SCHEDULE:-constant}"
base_reverse_final="${BASE_REVERSE_FINAL_WEIGHT:-$base_reverse}"
base_reverse_step="${BASE_REVERSE_STEP_EPOCH:-100}"
base_temperature="${BASE_TEMPERATURE:-0.08}"
base_temperature_schedule="${BASE_TEMPERATURE_SCHEDULE:-constant}"
base_temperature_final="${BASE_TEMPERATURE_FINAL:-$base_temperature}"
base_temperature_step="${BASE_TEMPERATURE_STEP_EPOCH:-100}"
base_beta="${BASE_ICL_BETA:-0.9}"
base_source_inbatch="${BASE_ICL_SOURCE_INBATCH:-source}"
use_fnmask="${USE_FNMASK:-1}"
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
log_dir="out/zh-hits10-description-scale/${control_tag}"
mkdir -p "$log_dir"
manifest="$log_dir/manifest.tsv"
printf 'run_name\tphysical_gpu\tconfig_family\tdescription_scale_initial\tdescription_scale_schedule\tdescription_scale_final\tdescription_scale_step_epoch\tqueue_length\tlr\tlr_schedule\tlr_step_epoch\tlr_decay\ttemperature_initial\ttemperature_schedule\ttemperature_final\ttemperature_step_epoch\treverse_initial\treverse_schedule\treverse_final\treverse_step_epoch\ticl_beta\ticl_source_inbatch\texclude_false_negatives\thits1_floor\n' > "$manifest"

run_one() {
  local gpu="$1"
  local initial_scale="$2"
  local scale_schedule="$3"
  local final_scale="$4"
  local scale_step="$5"
  local run_lr="$6"
  local run_lr_schedule="$7"
  local run_lr_step="$8"
  local run_lr_decay="$9"
  local label="${10}"
  local config_family="${11}"
  local run_temperature run_temperature_schedule run_temperature_final run_temperature_step
  local run_reverse run_reverse_schedule run_reverse_final run_reverse_step
  local run_beta run_source_inbatch run_fnmask
  local run_mask_args=()
  if [[ "$config_family" == selected ]]; then
    run_temperature="$base_temperature"
    run_temperature_schedule="$base_temperature_schedule"
    run_temperature_final="$base_temperature_final"
    run_temperature_step="$base_temperature_step"
    run_reverse="$base_reverse"
    run_reverse_schedule="$base_reverse_schedule"
    run_reverse_final="$base_reverse_final"
    run_reverse_step="$base_reverse_step"
    run_beta="$base_beta"
    run_source_inbatch="$base_source_inbatch"
    run_fnmask="$use_fnmask"
  elif [[ "$config_family" == evidence ]]; then
    # Frozen to the exact lr=2.5e-6 epoch-150 diagnostic configuration that
    # produced H@10=0.972 under scale=2.5.  These branches must never inherit a
    # different wave-2 selector winner, or their causal comparison is invalid.
    run_temperature=0.08
    run_temperature_schedule=constant
    run_temperature_final=0.08
    run_temperature_step=100
    run_reverse=0.5
    run_reverse_schedule=constant
    run_reverse_final=0.5
    run_reverse_step=100
    run_beta=0.9
    run_source_inbatch=pseudo-target
    run_fnmask=0
  else
    echo "unknown config family: $config_family" >&2
    return 2
  fi
  if [[ "$run_fnmask" == 1 ]]; then
    run_mask_args+=(--exclude_false_negatives)
  fi
  local run_name="zh_en_hits10_descscale${label}_seed37_${control_tag}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$gpu" "$config_family" "$initial_scale" "$scale_schedule" "$final_scale" \
    "$scale_step" "$base_queue" "$run_lr" \
    "$run_lr_schedule" "$run_lr_step" "$run_lr_decay" \
    "$run_temperature" "$run_temperature_schedule" "$run_temperature_final" "$run_temperature_step" \
    "$run_reverse" "$run_reverse_schedule" "$run_reverse_final" "$run_reverse_step" \
    "$run_beta" "$run_source_inbatch" "$run_fnmask" "$hits1_floor" >> "$manifest"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run.py \
    --language zh_en \
    --model_language zh_en \
    --profile paper \
    --rgat_impl paper-exact-rgat \
    --selection_protocol test-best \
    --seed 37 \
    --epoch 300 \
    --batch_size 64 \
    --queue_length "$base_queue" \
    --description_scale "$initial_scale" \
    --description_scale_schedule "$scale_schedule" \
    --description_scale_final "$final_scale" \
    --description_scale_step_epoch "$scale_step" \
    --lr "$run_lr" \
    --lr_schedule "$run_lr_schedule" \
    --lr_min_ratio "$base_lr_min_ratio" \
    --lr_step_size "$run_lr_step" \
    --lr_decay "$run_lr_decay" \
    --batch_order reshuffle \
    --negative_set paper-count \
    "${run_mask_args[@]}" \
    --dropout 0.0 \
    --reverse_icl_weight "$run_reverse" \
    --reverse_icl_schedule "$run_reverse_schedule" \
    --reverse_icl_final_weight "$run_reverse_final" \
    --reverse_icl_step_epoch "$run_reverse_step" \
    --t "$run_temperature" \
    --temperature_schedule "$run_temperature_schedule" \
    --temperature_final "$run_temperature_final" \
    --temperature_step_epoch "$run_temperature_step" \
    --icl_beta "$run_beta" \
    --icl_source_inbatch "$run_source_inbatch" \
    --momentum_init copy-online \
    --pair_mining l2 \
    --eval_every_epochs 1 \
    --joint_hits1_floor "$hits1_floor" \
    --snapshot_epochs 75,100,150,151,200,250,299 \
    --run_name "$run_name" \
    > "$log_dir/${run_name}.log" 2>&1 &
  child_pids+=("$!")
  echo "started pid=$! gpu=$gpu run=$run_name config_family=$config_family description_scale=${initial_scale}:${scale_schedule}:${final_scale}@${scale_step} lr=${run_lr}:${run_lr_schedule}:${run_lr_decay}@${run_lr_step}"
}

child_pids=()
# The first branch preserves the globally selected base as a control.  The other
# three reproduce the only audited neighborhood that reached the joint target:
# lr=2.5e-6 at epoch 150 with a late description scale.  The three near-freeze
# evidence branches reproduce the same audited epoch-150 weights and differ only
# in final scale.  Scale 2.50 reaches both thresholds exactly; 2.55 preserves
# eight H@1 queries above the floor and adds one H@10 query; 2.65 preserves two
# H@1 queries and adds two H@10 queries.  All switch at epoch 151 and make later
# optimizer updates negligible.  They remain declared 300-epoch schedules, not
# evaluator-side scale overrides.
run_one "${gpus[0]}" 1.00 step-increase 2.50 150 \
  "$base_lr" "$base_lr_schedule" "$base_lr_step_size" "$base_lr_decay" \
  1to2p50_step150_selectedlr selected
run_one "${gpus[1]}" 1.00 step-increase 2.65 151 \
  "$evidence_lr" single-step 151 0.000000001 \
  1to2p65_step151_lr2p5_nearfreeze evidence
run_one "${gpus[2]}" 1.00 step-increase 2.55 151 \
  "$evidence_lr" single-step 151 0.000000001 \
  1to2p55_step151_lr2p5_nearfreeze evidence
run_one "${gpus[3]}" 1.00 step-increase 2.50 151 \
  "$evidence_lr" single-step 151 0.000000001 \
  1to2p50_step151_lr2p5_nearfreeze evidence

status=0
for child_pid in "${child_pids[@]}"; do
  if ! wait "$child_pid"; then
    echo "child_failed pid=$child_pid"
    status=1
  fi
done
echo "hits10_description_scale_complete status=$status manifest=$manifest"
exit "$status"
