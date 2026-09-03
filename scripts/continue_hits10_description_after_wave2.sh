#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
parent_pid="${PARENT_PID:?PARENT_PID is required}"
wave1_tag="${WAVE1_TAG:-20260830_hits10_wave1}"
wave2_tag="${WAVE2_TAG:-20260830_hits10_wave2_temperature_beta}"
wave5_tag="${WAVE5_TAG:-20260830_hits10_wave5_description_scale}"
hits1_floor="${HITS1_FLOOR:-0.8887619047619048}"
hits10_target="${HITS10_TARGET:-0.972}"
audit_dir="out/zh-hits10/${wave5_tag}"
mkdir -p "$audit_dir" out/diagnostics

while kill -0 "$parent_pid" 2>/dev/null; do
  sleep 60
done

verify_selection() {
  local selection_path="$1"
  local phase="$2"
  local run_name checkpoint independent verification
  run_name="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_run"])' "$selection_path")"
  checkpoint="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["joint_checkpoint"])' "$selection_path")"
  independent="out/diagnostics/${run_name}_${phase}_strict_checkpoint_eval.json"
  verification="out/diagnostics/${run_name}_${phase}_joint_goal_verification.json"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" -u scripts/evaluate_checkpoints.py \
    --checkpoint "$checkpoint" \
    --candidate-scope full-target \
    --weight-source online \
    --device cuda \
    --output "$independent" \
    > "$audit_dir/${run_name}_${phase}_independent.log" 2>&1
  "$python_bin" scripts/verify_joint_goal.py \
    --run-name "$run_name" \
    --minimum-hits1 "$hits1_floor" \
    --minimum-hits10 "$hits10_target" \
    --checkpoint-evaluation "$independent" \
    > "$verification"
  echo "joint_goal_verified phase=$phase run=$run_name"
}

read_argument() {
  local selection_path="$1"
  local key="$2"
  local fallback="$3"
  "$python_bin" -c 'import json,sys; value=json.load(open(sys.argv[1]))["arguments"].get(sys.argv[2]); print(sys.argv[3] if value is None else value)' \
    "$selection_path" "$key" "$fallback"
}

base_selection="$audit_dir/base_selection.json"
"$python_bin" scripts/select_joint_candidate.py \
  --tag "$wave1_tag" \
  --tag "$wave2_tag" \
  --hits1-floor "$hits1_floor" \
  --hits10-target "$hits10_target" \
  --minimum-complete 8 \
  --required-run-manifest "out/zh-hits10/${wave1_tag}/manifest.tsv" \
  --required-run-manifest "out/zh-temperature-beta/${wave2_tag}/manifest.tsv" \
  --output "$base_selection"

target_reached="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["target_reached"]))' "$base_selection")"
if [[ "$target_reached" == 1 ]]; then
  verify_selection "$base_selection" pre_description
  exit 0
fi

base_queue="$(read_argument "$base_selection" queue_length 32)"
base_lr="$(read_argument "$base_selection" lr 2e-6)"
base_schedule="$(read_argument "$base_selection" lr_schedule constant)"
base_min_ratio="$(read_argument "$base_selection" lr_min_ratio 0.1)"
base_step_size="$(read_argument "$base_selection" lr_step_size 20)"
base_decay="$(read_argument "$base_selection" lr_decay 0.9)"
base_reverse="$(read_argument "$base_selection" reverse_icl_weight 0.5)"
base_reverse_schedule="$(read_argument "$base_selection" reverse_icl_schedule constant)"
base_reverse_final="$(read_argument "$base_selection" reverse_icl_final_weight "$base_reverse")"
base_reverse_step="$(read_argument "$base_selection" reverse_icl_step_epoch 100)"
base_temperature="$(read_argument "$base_selection" t 0.08)"
base_temperature_schedule="$(read_argument "$base_selection" temperature_schedule constant)"
base_temperature_final="$(read_argument "$base_selection" temperature_final "$base_temperature")"
base_temperature_step="$(read_argument "$base_selection" temperature_step_epoch 100)"
base_beta="$(read_argument "$base_selection" icl_beta 0.9)"
base_source_inbatch="$(read_argument "$base_selection" icl_source_inbatch pseudo-target)"
use_fnmask="$($python_bin -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1]))["arguments"].get("exclude_false_negatives", False))))' "$base_selection")"

GPU_IDS=0,1,2,3 \
CONTROL_TAG="$wave5_tag" \
BASE_QUEUE_LENGTH="$base_queue" \
BASE_LR="$base_lr" \
BASE_LR_SCHEDULE="$base_schedule" \
BASE_LR_MIN_RATIO="$base_min_ratio" \
BASE_LR_STEP_SIZE="$base_step_size" \
BASE_LR_DECAY="$base_decay" \
EVIDENCE_LR=2.5e-6 \
BASE_REVERSE_WEIGHT="$base_reverse" \
BASE_REVERSE_SCHEDULE="$base_reverse_schedule" \
BASE_REVERSE_FINAL_WEIGHT="$base_reverse_final" \
BASE_REVERSE_STEP_EPOCH="$base_reverse_step" \
BASE_TEMPERATURE="$base_temperature" \
BASE_TEMPERATURE_SCHEDULE="$base_temperature_schedule" \
BASE_TEMPERATURE_FINAL="$base_temperature_final" \
BASE_TEMPERATURE_STEP_EPOCH="$base_temperature_step" \
BASE_ICL_BETA="$base_beta" \
BASE_ICL_SOURCE_INBATCH="$base_source_inbatch" \
USE_FNMASK="$use_fnmask" \
HITS1_FLOOR="$hits1_floor" \
  bash scripts/launch_zh_hits10_description_scale.sh

wave5_selection="$audit_dir/wave5_selection.json"
"$python_bin" scripts/select_joint_candidate.py \
  --tag "$wave1_tag" \
  --tag "$wave2_tag" \
  --tag "$wave5_tag" \
  --hits1-floor "$hits1_floor" \
  --hits10-target "$hits10_target" \
  --minimum-complete 12 \
  --required-run-manifest "out/zh-hits10/${wave1_tag}/manifest.tsv" \
  --required-run-manifest "out/zh-temperature-beta/${wave2_tag}/manifest.tsv" \
  --required-run-manifest "out/zh-hits10-description-scale/${wave5_tag}/manifest.tsv" \
  --output "$wave5_selection"

target_reached="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["target_reached"]))' "$wave5_selection")"
if [[ "$target_reached" == 1 ]]; then
  verify_selection "$wave5_selection" description
else
  echo "joint_goal_not_yet_reached_after_description selection=$wave5_selection"
fi
