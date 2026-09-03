#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
wave2_pid="${WAVE2_PID:?WAVE2_PID is required}"
wave2_tag="${WAVE2_TAG:-20260830_goal88_wave2}"
wave3_tag="${WAVE3_TAG:-20260830_goal88_wave3_fnmask}"

while kill -0 "$wave2_pid" 2>/dev/null; do
  sleep 60
done

"$python_bin" - "$wave2_tag" <<'PY'
import json
import sys
from pathlib import Path

tag = sys.argv[1]
names = (
    'zh_en_exact_dropout0_paper_count_q32_seed37_' + tag,
    'zh_en_exact_reverse05_paper_count_q32_seed37_' + tag,
    'zh_en_exact_dropout0_reverse05_paper_count_q32_seed37_' + tag,
    'zh_en_exact_dropout0_reverse05_lr2e6_paper_count_q32_seed37_' + tag,
)
for name in names:
    path = Path('out/results') / (name + '.json')
    if not path.is_file():
        raise SystemExit('missing wave2 result: {}'.format(path))
    payload = json.loads(path.read_text())
    if payload.get('status') != 'complete' or len(payload.get('test_history', ())) != 300:
        raise SystemExit('incomplete wave2 result: {}'.format(path))
print('wave2_result_gate=passed')
PY

mkdir -p out/zh-false-negative
CUDA_VISIBLE_DEVICES=0 "$python_bin" -u run.py \
  --language zh_en \
  --profile paper \
  --rgat_impl paper-exact-rgat \
  --epoch 1 \
  --max_steps 50 \
  --queue_length 32 \
  --lr_schedule constant \
  --negative_set paper-count \
  --exclude_false_negatives \
  --dropout 0 \
  --reverse_icl_weight 0.5 \
  --run_name "smoke_fnmask_dropout0_reverse05_${wave3_tag}" \
  > "out/zh-false-negative/${wave3_tag}_smoke.log" 2>&1

GPU_IDS=0,1,2,3 CONTROL_TAG="$wave3_tag" \
  bash scripts/launch_zh_false_negative.sh
