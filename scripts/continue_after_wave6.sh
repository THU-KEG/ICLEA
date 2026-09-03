#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
high_lr_pid="${HIGH_LR_PID:?HIGH_LR_PID is required}"
remaining_pid="${REMAINING_PID:?REMAINING_PID is required}"
wave7_tag="${WAVE7_TAG:-20260830_goal88_wave7_temperature_beta}"

for watched_pid in "$high_lr_pid" "$remaining_pid"; do
  while kill -0 "$watched_pid" 2>/dev/null; do
    sleep 60
  done
done

audit_dir="out/zh-temperature-beta/${wave7_tag}"
selection="${audit_dir}/completed_selection.tsv"
mkdir -p "$audit_dir" out/diagnostics
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
    raise SystemExit('no complete goal88 result found after wave6')
best = max(complete, key=lambda payload: payload['best_test']['hits1'])
arguments = best.get('arguments', {})
output.write_text(
    'run_name\thits1\tcheckpoint\tlr\treverse\tfnmask\n'
    '{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        best['run_name'],
        best['best_test']['hits1'],
        best['checkpoint'],
        arguments.get('lr', 2e-6),
        arguments.get('reverse_icl_weight', 0.0),
        int(arguments.get('exclude_false_negatives', False)),
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
IFS=$'\t' read -r best_run best_hits1 checkpoint base_lr base_reverse use_fnmask \
  < <(tail -n 1 "$selection")

if [[ "$selection_status" -eq 88 ]]; then
  report="out/diagnostics/${best_run}_strict_checkpoint_eval.json"
  verification="out/diagnostics/${best_run}_goal88_verification.json"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" -u scripts/evaluate_checkpoints.py \
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

mask_args=()
if [[ "$use_fnmask" == 1 ]]; then
  mask_args+=(--exclude_false_negatives)
fi
CUDA_VISIBLE_DEVICES=0 "$python_bin" -u run.py \
  --language zh_en \
  --profile paper \
  --rgat_impl paper-exact-rgat \
  --epoch 1 \
  --max_steps 50 \
  --queue_length 32 \
  --lr "$base_lr" \
  --lr_schedule constant \
  --negative_set paper-count \
  "${mask_args[@]}" \
  --dropout 0 \
  --reverse_icl_weight "$base_reverse" \
  --t 0.06 \
  --icl_beta 0.9 \
  --run_name "smoke_temperature_beta_${wave7_tag}" \
  > "${audit_dir}/smoke.log" 2>&1

GPU_IDS=0,1,2,3 CONTROL_TAG="$wave7_tag" BASE_LR="$base_lr" \
BASE_REVERSE_WEIGHT="$base_reverse" USE_FNMASK="$use_fnmask" \
  bash scripts/launch_zh_temperature_beta.sh
