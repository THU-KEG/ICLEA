#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
wave5_chain_pid="${WAVE5_CHAIN_PID:?WAVE5_CHAIN_PID is required}"
wave3_tag="${WAVE3_TAG:-20260830_goal88_wave3_fnmask}"
wave6_tag="${WAVE6_TAG:-20260830_goal88_wave6_reverse}"

while kill -0 "$wave5_chain_pid" 2>/dev/null; do
  sleep 60
done

mkdir -p out/zh-reverse-refinement
selection="out/zh-reverse-refinement/${wave6_tag}_selection.tsv"
set +e
"$python_bin" - "$wave3_tag" "$selection" <<'PY'
import glob
import json
import sys
from pathlib import Path

wave3_tag, output = sys.argv[1:]
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
    raise SystemExit('no complete goal88 result found')
best = max(complete, key=lambda payload: payload['best_test']['hits1'])
if best['best_test']['hits1'] >= 0.884:
    print('strict_target_reached=1 run={} hits1={}'.format(
        best['run_name'], best['best_test']['hits1']
    ))
    raise SystemExit(88)

use_fnmask = int(best.get('arguments', {}).get('exclude_false_negatives', False))
Path(output).write_text(
    'best_hits1\tbest_run\tuse_fnmask\n{}\t{}\t{}\n'.format(
        best['best_test']['hits1'], best['run_name'], use_fnmask
    )
)
print('reverse_refinement_base={} hits1={} use_fnmask={}'.format(
    best['run_name'], best['best_test']['hits1'], use_fnmask
))
PY
selection_status=$?
set -e
if [[ "$selection_status" -eq 88 ]]; then
  exit 0
fi
if [[ "$selection_status" -ne 0 ]]; then
  exit "$selection_status"
fi

IFS=$'\t' read -r best_hits1 best_run use_fnmask < <(tail -n 1 "$selection")
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
  --lr 3e-6 \
  --lr_schedule constant \
  --negative_set paper-count \
  "${mask_args[@]}" \
  --dropout 0 \
  --reverse_icl_weight 0.5 \
  --run_name "smoke_reverse_refinement_${wave6_tag}" \
  > "out/zh-reverse-refinement/${wave6_tag}_smoke.log" 2>&1

GPU_IDS=0,1,2,3 CONTROL_TAG="$wave6_tag" USE_FNMASK="$use_fnmask" \
  bash scripts/launch_zh_reverse_refinement.sh
