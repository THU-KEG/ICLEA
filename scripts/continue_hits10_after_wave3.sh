#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
parent_pid="${PARENT_PID:?PARENT_PID is required}"
wave1_tag="${WAVE1_TAG:-20260830_hits10_wave1}"
wave2_tag="${WAVE2_TAG:-20260830_hits10_wave2_temperature_beta}"
wave3_tag="${WAVE3_TAG:-20260830_hits10_wave3_queue_lrstep}"
wave4_tag="${WAVE4_TAG:-20260830_hits10_wave4_late_relaxation}"
hits1_floor="${HITS1_FLOOR:-0.8887619047619048}"
hits10_target="${HITS10_TARGET:-0.972}"
audit_dir="out/zh-hits10/${wave4_tag}"
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
  if [[ -f "$verification" ]]; then
    echo "joint_goal_already_verified phase=$phase run=$run_name"
    return
  fi
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

base_selection="out/zh-hits10/${wave3_tag}/base_selection.json"
wave3_selection="out/zh-hits10/${wave3_tag}/wave3_selection.json"
for prior in "$wave3_selection" "$base_selection"; do
  if [[ -f "$prior" ]]; then
    prior_target="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("target_reached", False)))' "$prior")"
    if [[ "$prior_target" == 1 ]]; then
      verify_selection "$prior" prior
      exit 0
    fi
  fi
done
if [[ ! -f "$wave3_selection" ]]; then
  echo "missing completed wave3 selection: $wave3_selection" >&2
  exit 1
fi

selection="$audit_dir/base_selection.json"
"$python_bin" scripts/select_joint_candidate.py \
  --tag "$wave1_tag" \
  --tag "$wave2_tag" \
  --tag "$wave3_tag" \
  --hits1-floor "$hits1_floor" \
  --hits10-target "$hits10_target" \
  --minimum-complete 12 \
  --required-run-manifest "out/zh-hits10/${wave1_tag}/manifest.tsv" \
  --required-run-manifest "out/zh-temperature-beta/${wave2_tag}/manifest.tsv" \
  --required-run-manifest "out/zh-hits10-refinement/${wave3_tag}/manifest.tsv" \
  --output "$selection"

target_reached="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["target_reached"]))' "$selection")"
if [[ "$target_reached" == 1 ]]; then
  verify_selection "$selection" wave3_global
  exit 0
fi

read_argument() {
  local key="$1"
  local fallback="$2"
  "$python_bin" -c 'import json,sys; value=json.load(open(sys.argv[1]))["arguments"].get(sys.argv[2]); print(sys.argv[3] if value is None else value)' "$selection" "$key" "$fallback"
}

base_queue="$(read_argument queue_length 32)"
base_lr="$(read_argument lr 2e-6)"
base_schedule="$(read_argument lr_schedule constant)"
base_min_ratio="$(read_argument lr_min_ratio 0.1)"
base_step_size="$(read_argument lr_step_size 20)"
base_decay="$(read_argument lr_decay 0.9)"
base_reverse="$(read_argument reverse_icl_weight 0.5)"
base_temperature="$(read_argument t 0.08)"
base_beta="$(read_argument icl_beta 0.9)"
base_source_inbatch="$(read_argument icl_source_inbatch pseudo-target)"
use_fnmask="$($python_bin -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1]))["arguments"].get("exclude_false_negatives", False))))' "$selection")"
final_temperature="$($python_bin -c 'import sys; value=float(sys.argv[1]); print(0.10 if value < 0.10 else 0.12)' "$base_temperature")"

GPU_IDS=0,1,2,3 \
CONTROL_TAG="$wave4_tag" \
BASE_QUEUE_LENGTH="$base_queue" \
BASE_LR="$base_lr" \
BASE_LR_SCHEDULE="$base_schedule" \
BASE_LR_MIN_RATIO="$base_min_ratio" \
BASE_LR_STEP_SIZE="$base_step_size" \
BASE_LR_DECAY="$base_decay" \
BASE_REVERSE_WEIGHT="$base_reverse" \
BASE_TEMPERATURE="$base_temperature" \
FINAL_TEMPERATURE="$final_temperature" \
BASE_ICL_BETA="$base_beta" \
BASE_ICL_SOURCE_INBATCH="$base_source_inbatch" \
USE_FNMASK="$use_fnmask" \
HITS1_FLOOR="$hits1_floor" \
  bash scripts/launch_zh_hits10_late_relaxation.sh

wave4_selection="$audit_dir/wave4_selection.json"
"$python_bin" scripts/select_joint_candidate.py \
  --tag "$wave1_tag" \
  --tag "$wave2_tag" \
  --tag "$wave3_tag" \
  --tag "$wave4_tag" \
  --hits1-floor "$hits1_floor" \
  --hits10-target "$hits10_target" \
  --minimum-complete 16 \
  --required-run-manifest "out/zh-hits10/${wave1_tag}/manifest.tsv" \
  --required-run-manifest "out/zh-temperature-beta/${wave2_tag}/manifest.tsv" \
  --required-run-manifest "out/zh-hits10-refinement/${wave3_tag}/manifest.tsv" \
  --required-run-manifest "out/zh-hits10-late-relaxation/${wave4_tag}/manifest.tsv" \
  --output "$wave4_selection"

target_reached="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["target_reached"]))' "$wave4_selection")"
if [[ "$target_reached" == 1 ]]; then
  verify_selection "$wave4_selection" wave4
else
  echo "joint_goal_not_yet_reached_after_wave4 selection=$wave4_selection"
fi
