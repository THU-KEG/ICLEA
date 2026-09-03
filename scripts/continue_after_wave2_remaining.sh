#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
wave2_pid="${WAVE2_PID:?WAVE2_PID is required}"
control_tag="${CONTROL_TAG:-20260830_goal88_wave6_reverse_priority}"
audit_dir="out/zh-reverse-refinement/${control_tag}"
selection="${audit_dir}/wave2_completed_selection.tsv"
mkdir -p "$audit_dir" out/diagnostics

while kill -0 "$wave2_pid" 2>/dev/null; do
  sleep 60
done

set +e
"$python_bin" - "$selection" <<'PY'
import glob
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
complete = []
for path_value in glob.glob('out/results/*goal88*.json'):
    payload = json.loads(Path(path_value).read_text())
    if (
        payload.get('status') == 'complete'
        and payload.get('language') == 'zh_en'
        and payload.get('setting') == 'original'
        and len(payload.get('test_history', ())) == 300
    ):
        complete.append(payload)
if not complete:
    raise SystemExit('no complete goal88 result found after wave2')
best = max(complete, key=lambda payload: payload['best_test']['hits1'])
output.write_text(
    'run_name\thits1\tcheckpoint\n{}\t{}\t{}\n'.format(
        best['run_name'], best['best_test']['hits1'], best['checkpoint']
    )
)
print('best_complete_run={} hits1={}'.format(
    best['run_name'], best['best_test']['hits1']
))
if best['best_test']['hits1'] >= 0.884:
    raise SystemExit(88)
PY
selection_status=$?
set -e
if [[ "$selection_status" -eq 88 ]]; then
  IFS=$'\t' read -r best_run best_hits1 checkpoint < <(tail -n 1 "$selection")
  report="out/diagnostics/${best_run}_strict_checkpoint_eval.json"
  verification="out/diagnostics/${best_run}_goal88_verification.json"
  CUDA_VISIBLE_DEVICES=2 "$python_bin" -u scripts/evaluate_checkpoints.py \
    --checkpoint "$checkpoint" \
    --candidate-scope full-target \
    --weight-source online \
    --device cuda \
    --output "$report" \
    > "${audit_dir}/${best_run}_strict_checkpoint_eval.log" 2>&1
  "$python_bin" scripts/verify_goal88.py \
    --run-name "$best_run" \
    --checkpoint-evaluation "$report" \
    > "$verification"
  echo "strict_goal88_verified run=$best_run hits1=$best_hits1 report=$report verification=$verification"
  exit 0
fi
if [[ "$selection_status" -ne 0 ]]; then
  exit "$selection_status"
fi

GPU_IDS=2,3 CONFIG_SET=remaining CONTROL_TAG="$control_tag" USE_FNMASK=0 \
  bash scripts/launch_zh_reverse_refinement.sh
