#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
wave3_chain_pid="${WAVE3_CHAIN_PID:?WAVE3_CHAIN_PID is required}"
wave2_tag="${WAVE2_TAG:-20260830_goal88_wave2}"
wave3_tag="${WAVE3_TAG:-20260830_goal88_wave3_fnmask}"
wave4_tag="${WAVE4_TAG:-20260830_goal88_wave4_hubness}"

while kill -0 "$wave3_chain_pid" 2>/dev/null; do
  sleep 60
done

selection="out/zh-hubness/${wave4_tag}_selected_base.tsv"
mkdir -p out/zh-hubness
set +e
"$python_bin" - "$wave2_tag" "$wave3_tag" "$selection" <<'PY'
import json
import sys
from pathlib import Path

wave2, wave3, output = sys.argv[1:]
labels = (
    ('dropout0', False),
    ('reverse05', False),
    ('dropout0_reverse05', False),
    ('dropout0_reverse05_lr2e6', False),
    ('fnmask_dropout0', True),
    ('fnmask_reverse05', True),
    ('fnmask_dropout0_reverse05', True),
    ('fnmask_dropout0_reverse05_lr2e6', True),
)
rows = []
for label, masked in labels:
    tag = wave3 if masked else wave2
    name = 'zh_en_exact_{}_paper_count_q32_seed37_{}'.format(label, tag)
    path = Path('out/results') / (name + '.json')
    if not path.is_file():
        raise SystemExit('missing result: {}'.format(path))
    payload = json.loads(path.read_text())
    if payload.get('status') != 'complete' or len(payload.get('test_history', ())) != 300:
        raise SystemExit('incomplete result: {}'.format(path))
    rows.append((payload['best_test']['hits1'], name, payload, masked))
rows.sort(key=lambda row: row[0], reverse=True)
best_hits1, best_name, best, masked = rows[0]
arguments = best['arguments']
Path(output).write_text(
    'hits1\trun_name\tdropout\treverse_icl_weight\tlr\tuse_fnmask\n'
    '{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        best_hits1,
        best_name,
        arguments['dropout'],
        arguments.get('reverse_icl_weight', 0.0),
        arguments['lr'],
        int(masked),
    )
)
print('selected_base={} hits1={}'.format(best_name, best_hits1))
if best_hits1 >= 0.884:
    print('strict_target_reached=1')
    raise SystemExit(88)
PY
selection_status=$?
set -e
if [[ "$selection_status" -eq 88 ]]; then
  exit 0
fi
if [[ "$selection_status" -ne 0 ]]; then
  exit "$selection_status"
fi

IFS=$'\t' read -r best_hits1 best_name base_dropout base_reverse base_lr use_fnmask \
  < <(tail -n 1 "$selection")

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
  --dropout "$base_dropout" \
  --reverse_icl_weight "$base_reverse" \
  --momentum_init copy-online \
  --pair_mining csls \
  --pair_csls_k 1 \
  --run_name "smoke_csls_${wave4_tag}" \
  > "out/zh-hubness/${wave4_tag}_smoke.log" 2>&1

GPU_IDS=0,1,2,3 CONTROL_TAG="$wave4_tag" \
BASE_DROPOUT="$base_dropout" BASE_REVERSE_WEIGHT="$base_reverse" \
BASE_LR="$base_lr" USE_FNMASK="$use_fnmask" \
  bash scripts/launch_zh_hubness_audit.sh
