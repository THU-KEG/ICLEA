#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
wave1_pid="${WAVE1_PID:?WAVE1_PID is required}"
wave1_tag="${WAVE1_TAG:-20260830_hits10_wave1}"
wave2_tag="${WAVE2_TAG:-20260830_hits10_wave2_temperature_beta}"
hits1_floor="${HITS1_FLOOR:-0.8887619047619048}"
hits10_target="${HITS10_TARGET:-0.972}"
audit_dir="out/zh-hits10/${wave2_tag}"
mkdir -p "$audit_dir" out/diagnostics

while kill -0 "$wave1_pid" 2>/dev/null; do
  sleep 60
done

selection="$audit_dir/wave1_selection.json"
set +e
"$python_bin" - "$wave1_tag" "$hits1_floor" "$hits10_target" "$selection" <<'PY'
import glob
import json
import sys
from pathlib import Path

tag, hits1_floor, hits10_target, output = sys.argv[1:]
hits1_floor = float(hits1_floor)
hits10_target = float(hits10_target)
paths = sorted(glob.glob('out/results/zh_en_hits10_*_{}.json'.format(tag)))
if len(paths) != 4:
    raise SystemExit('expected four wave1 results, found {}'.format(len(paths)))
payloads = [json.loads(Path(path).read_text()) for path in paths]
if any(p.get('status') != 'complete' or len(p.get('test_history', ())) != 300 for p in payloads):
    raise SystemExit('one or more wave1 runs are incomplete')
eligible = [
    p for p in payloads
    if p.get('best_joint_test')
    and p['best_joint_test']['hits1'] >= hits1_floor
]
best = max(
    eligible or payloads,
    key=lambda p: (
        (p.get('best_joint_test') or {}).get('hits10', -1.0),
        (p.get('best_joint_test') or p.get('best_test', {})).get('hits1', -1.0),
        p.get('best_test', {}).get('hits1', -1.0),
    ),
)
selected = best.get('best_joint_test') or best.get('best_test', {})
report = {
    'wave1_tag': tag,
    'runs': [{
        'run_name': p['run_name'],
        'best_test': p.get('best_test'),
        'best_joint_test': p.get('best_joint_test'),
        'joint_checkpoint': p.get('joint_checkpoint'),
        'arguments': {
            key: p.get('arguments', {}).get(key)
            for key in ('lr', 'lr_schedule', 'lr_min_ratio', 'reverse_icl_weight', 't', 'icl_beta')
        },
    } for p in payloads],
    'selected_run': best['run_name'],
    'selected_metrics': selected,
    'joint_checkpoint': best.get('joint_checkpoint'),
    'target_reached': bool(
        best.get('joint_checkpoint')
        and selected.get('hits1', -1.0) >= hits1_floor
        and selected.get('hits10', -1.0) >= hits10_target
    ),
    'next_base': {
        'lr': best.get('arguments', {}).get('lr', 2e-6),
        'lr_schedule': best.get('arguments', {}).get('lr_schedule', 'constant'),
        'lr_min_ratio': best.get('arguments', {}).get('lr_min_ratio', 0.1),
        'reverse_icl_weight': best.get('arguments', {}).get('reverse_icl_weight', 0.5),
    },
}
Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, sort_keys=True))
raise SystemExit(88 if report['target_reached'] else 0)
PY
selection_status=$?
set -e

if [[ "$selection_status" -eq 88 ]]; then
  run_name="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_run"])' "$selection")"
  checkpoint="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["joint_checkpoint"])' "$selection")"
  independent="out/diagnostics/${run_name}_joint_strict_checkpoint_eval.json"
  verification="out/diagnostics/${run_name}_joint_goal_verification.json"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" -u scripts/evaluate_checkpoints.py \
    --checkpoint "$checkpoint" \
    --candidate-scope full-target \
    --weight-source online \
    --device cuda \
    --output "$independent" \
    > "$audit_dir/${run_name}_independent.log" 2>&1
  "$python_bin" scripts/verify_joint_goal.py \
    --run-name "$run_name" \
    --minimum-hits1 "$hits1_floor" \
    --minimum-hits10 "$hits10_target" \
    --checkpoint-evaluation "$independent" \
    > "$verification"
  echo "joint_goal_verified_after_wave1 run=$run_name"
  exit 0
fi
if [[ "$selection_status" -ne 0 ]]; then
  exit "$selection_status"
fi

base_lr="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_base"]["lr"])' "$selection")"
base_schedule="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_base"]["lr_schedule"])' "$selection")"
base_min_ratio="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_base"]["lr_min_ratio"])' "$selection")"
base_reverse="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_base"]["reverse_icl_weight"])' "$selection")"

GPU_IDS=0,1,2,3 \
CONTROL_TAG="$wave2_tag" \
BASE_LR="$base_lr" \
BASE_LR_SCHEDULE="$base_schedule" \
BASE_LR_MIN_RATIO="$base_min_ratio" \
BASE_REVERSE_WEIGHT="$base_reverse" \
HITS1_FLOOR="$hits1_floor" \
  bash scripts/launch_zh_temperature_beta.sh

"$python_bin" - "$wave2_tag" "$hits1_floor" "$hits10_target" "$audit_dir/wave2_summary.json" <<'PY'
import glob
import json
import sys
from pathlib import Path

tag, hits1_floor, hits10_target, output = sys.argv[1:]
hits1_floor = float(hits1_floor)
hits10_target = float(hits10_target)
payloads = [
    json.loads(Path(path).read_text())
    for path in sorted(glob.glob('out/results/zh_en_exact_*_{}.json'.format(tag)))
]
if len(payloads) != 4 or any(
    p.get('status') != 'complete' or len(p.get('test_history', ())) != 300
    for p in payloads
):
    raise SystemExit('wave2 did not produce four complete results')
eligible = [p for p in payloads if p.get('best_joint_test')]
best = max(
    eligible or payloads,
    key=lambda p: (
        (p.get('best_joint_test') or {}).get('hits10', -1.0),
        (p.get('best_joint_test') or p.get('best_test', {})).get('hits1', -1.0),
    ),
)
selected = best.get('best_joint_test') or best.get('best_test', {})
report = {
    'wave2_tag': tag,
    'selected_run': best['run_name'],
    'selected_metrics': selected,
    'joint_checkpoint': best.get('joint_checkpoint'),
    'target_reached': bool(
        best.get('joint_checkpoint')
        and selected.get('hits1', -1.0) >= hits1_floor
        and selected.get('hits10', -1.0) >= hits10_target
    ),
    'runs': [{
        'run_name': p['run_name'],
        'best_test': p.get('best_test'),
        'best_joint_test': p.get('best_joint_test'),
    } for p in payloads],
}
Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, sort_keys=True))
PY
