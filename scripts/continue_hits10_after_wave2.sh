#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
parent_pid="${PARENT_PID:?PARENT_PID is required}"
wave1_tag="${WAVE1_TAG:-20260830_hits10_wave1}"
wave2_tag="${WAVE2_TAG:-20260830_hits10_wave2_temperature_beta}"
wave3_tag="${WAVE3_TAG:-20260830_hits10_wave3_queue_lrstep}"
hits1_floor="${HITS1_FLOOR:-0.8887619047619048}"
hits10_target="${HITS10_TARGET:-0.972}"
audit_dir="out/zh-hits10/${wave3_tag}"
mkdir -p "$audit_dir" out/diagnostics

while kill -0 "$parent_pid" 2>/dev/null; do
  sleep 60
done

wave2_summary="out/zh-hits10/${wave2_tag}/wave2_summary.json"
if [[ ! -f "$wave2_summary" ]]; then
  wave1_selection="out/zh-hits10/${wave2_tag}/wave1_selection.json"
  if [[ -f "$wave1_selection" ]]; then
    wave1_target="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("target_reached", False)))' "$wave1_selection")"
    wave1_run="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("selected_run", ""))' "$wave1_selection")"
    if [[ "$wave1_target" == 1 && -f "out/diagnostics/${wave1_run}_joint_goal_verification.json" ]]; then
      echo "joint_goal_already_verified_before_wave3 run=$wave1_run"
      exit 0
    fi
  fi
  echo "missing completed wave2 summary: $wave2_summary" >&2
  exit 1
fi

selection="$audit_dir/base_selection.json"
"$python_bin" - "$wave1_tag" "$wave2_tag" "$hits1_floor" "$hits10_target" "$selection" <<'PY'
import glob
import json
import sys
from pathlib import Path

wave1, wave2, hits1_floor, hits10_target, output = sys.argv[1:]
hits1_floor = float(hits1_floor)
hits10_target = float(hits10_target)
paths = sorted(glob.glob('out/results/*_{}.json'.format(wave1)))
paths += sorted(glob.glob('out/results/*_{}.json'.format(wave2)))
complete = [json.loads(Path(path).read_text()) for path in paths]
complete = [
    payload for payload in complete
    if payload.get('status') == 'complete'
    and len(payload.get('test_history', ())) == 300
]
if len(complete) < 8:
    raise SystemExit('expected at least eight complete wave1/wave2 results, found {}'.format(len(complete)))
payloads = [payload for payload in complete if payload.get('best_joint_test')]
if not payloads:
    raise SystemExit('wave1/wave2 produced no checkpoint meeting the Hits@1 floor')
best = max(
    payloads,
    key=lambda payload: (
        payload['best_joint_test']['hits10'],
        payload['best_joint_test']['hits1'],
        -payload['best_joint_epoch'],
    ),
)
arguments = best['arguments']
metrics = best['best_joint_test']
report = {
    'selected_run': best['run_name'],
    'selected_metrics': metrics,
    'joint_checkpoint': best.get('joint_checkpoint'),
    'target_reached': bool(
        best.get('joint_checkpoint')
        and metrics['hits1'] >= hits1_floor
        and metrics['hits10'] >= hits10_target
    ),
    'arguments': {
        key: arguments.get(key)
        for key in (
            'lr', 'lr_schedule', 'lr_min_ratio', 'lr_step_size', 'lr_decay',
            'reverse_icl_weight', 't', 'icl_beta', 'icl_source_inbatch',
            'exclude_false_negatives',
        )
    },
    'eligible_runs': len(payloads),
}
Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, sort_keys=True))
PY

verify_selected() {
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

target_reached="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["target_reached"]))' "$selection")"
if [[ "$target_reached" == 1 ]]; then
  verify_selected "$selection" wave2
  exit 0
fi

base_lr="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"]["lr"])' "$selection")"
base_schedule="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"]["lr_schedule"])' "$selection")"
base_min_ratio="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"].get("lr_min_ratio", 0.1))' "$selection")"
base_step_size="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"].get("lr_step_size", 20))' "$selection")"
base_decay="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"].get("lr_decay", 0.9))' "$selection")"
base_reverse="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"]["reverse_icl_weight"])' "$selection")"
base_temperature="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"]["t"])' "$selection")"
base_beta="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"]["icl_beta"])' "$selection")"
base_source_inbatch="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["arguments"].get("icl_source_inbatch", "pseudo-target"))' "$selection")"
use_fnmask="$($python_bin -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1]))["arguments"].get("exclude_false_negatives", False))))' "$selection")"

GPU_IDS=0,1,2,3 \
CONTROL_TAG="$wave3_tag" \
BASE_LR="$base_lr" \
BASE_LR_SCHEDULE="$base_schedule" \
BASE_LR_MIN_RATIO="$base_min_ratio" \
BASE_LR_STEP_SIZE="$base_step_size" \
BASE_LR_DECAY="$base_decay" \
BASE_REVERSE_WEIGHT="$base_reverse" \
BASE_TEMPERATURE="$base_temperature" \
BASE_ICL_BETA="$base_beta" \
BASE_ICL_SOURCE_INBATCH="$base_source_inbatch" \
USE_FNMASK="$use_fnmask" \
HITS1_FLOOR="$hits1_floor" \
  bash scripts/launch_zh_hits10_refinement.sh

wave3_selection="$audit_dir/wave3_selection.json"
"$python_bin" - "$wave3_tag" "$hits1_floor" "$hits10_target" "$wave3_selection" <<'PY'
import glob
import json
import sys
from pathlib import Path

tag, hits1_floor, hits10_target, output = sys.argv[1:]
hits1_floor = float(hits1_floor)
hits10_target = float(hits10_target)
paths = sorted(glob.glob('out/results/zh_en_hits10_refine_*_{}.json'.format(tag)))
payloads = [json.loads(Path(path).read_text()) for path in paths]
if len(payloads) != 4 or any(
    payload.get('status') != 'complete'
    or len(payload.get('test_history', ())) != 300
    for payload in payloads
):
    raise SystemExit('wave3 did not produce four complete results')
eligible = [payload for payload in payloads if payload.get('best_joint_test')]
if not eligible:
    raise SystemExit('wave3 produced no checkpoint meeting the Hits@1 floor')
best = max(
    eligible,
    key=lambda payload: (
        payload['best_joint_test']['hits10'],
        payload['best_joint_test']['hits1'],
        -payload['best_joint_epoch'],
    ),
)
metrics = best['best_joint_test']
report = {
    'selected_run': best['run_name'],
    'selected_metrics': metrics,
    'joint_checkpoint': best.get('joint_checkpoint'),
    'target_reached': bool(
        best.get('joint_checkpoint')
        and metrics['hits1'] >= hits1_floor
        and metrics['hits10'] >= hits10_target
    ),
    'runs': [
        {
            'run_name': payload['run_name'],
            'best_test': payload.get('best_test'),
            'best_joint_test': payload.get('best_joint_test'),
        }
        for payload in payloads
    ],
}
Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, sort_keys=True))
PY

target_reached="$($python_bin -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["target_reached"]))' "$wave3_selection")"
if [[ "$target_reached" == 1 ]]; then
  verify_selected "$wave3_selection" wave3
else
  echo "joint_goal_not_yet_reached_after_wave3 selection=$wave3_selection"
fi
