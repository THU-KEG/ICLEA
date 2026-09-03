#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
wave4_chain_pid="${WAVE4_CHAIN_PID:?WAVE4_CHAIN_PID is required}"
wave4_tag="${WAVE4_TAG:-20260830_goal88_wave4_hubness}"
wave5_tag="${WAVE5_TAG:-20260830_goal88_wave5_lr}"

while kill -0 "$wave4_chain_pid" 2>/dev/null; do
  sleep 60
done

selection="out/zh-lr-refinement/${wave5_tag}_selected_base.tsv"
mkdir -p out/zh-lr-refinement
set +e
"$python_bin" - "$wave4_tag" "$selection" <<'PY'
import glob
import json
import sys
from pathlib import Path

wave4_tag, output = sys.argv[1:]
names = (
    'zh_en_exact_independent_l2_paperloss_paper_count_q32_seed37_' + wave4_tag,
    'zh_en_exact_copy_csls_k1_paperloss_paper_count_q32_seed37_' + wave4_tag,
    'zh_en_exact_copy_csls_k3_paperloss_paper_count_q32_seed37_' + wave4_tag,
    'zh_en_exact_independent_csls_k1_base_paper_count_q32_seed37_' + wave4_tag,
)
wave4 = []
for name in names:
    path = Path('out/results') / (name + '.json')
    if not path.is_file():
        raise SystemExit('missing wave4 result: {}'.format(path))
    payload = json.loads(path.read_text())
    if payload.get('status') != 'complete' or len(payload.get('test_history', ())) != 300:
        raise SystemExit('incomplete wave4 result: {}'.format(path))
    wave4.append(payload)

all_complete = []
for path_value in glob.glob('out/results/*goal88*.json'):
    payload = json.loads(Path(path_value).read_text())
    if payload.get('status') == 'complete' and len(payload.get('test_history', ())) == 300:
        all_complete.append(payload)
best_all = max(all_complete, key=lambda payload: payload['best_test']['hits1'])
if best_all['best_test']['hits1'] >= 0.884:
    print('strict_target_reached=1 run={} hits1={}'.format(
        best_all['run_name'], best_all['best_test']['hits1']
    ))
    raise SystemExit(88)

best_wave4 = max(wave4, key=lambda payload: payload['best_test']['hits1'])
arguments = best_wave4['arguments']
Path(output).write_text(
    'hits1\trun_name\tdropout\tuse_fnmask\n{}\t{}\t{}\t{}\n'.format(
        best_wave4['best_test']['hits1'],
        best_wave4['run_name'],
        arguments['dropout'],
        int(arguments.get('exclude_false_negatives', False)),
    )
)
print('selected_wave4_base={} hits1={}'.format(
    best_wave4['run_name'], best_wave4['best_test']['hits1']
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

IFS=$'\t' read -r best_hits1 best_name dropout use_fnmask < <(tail -n 1 "$selection")

CUDA_VISIBLE_DEVICES=0 "$python_bin" -u run.py \
  --language zh_en \
  --profile paper \
  --rgat_impl paper-exact-rgat \
  --epoch 1 \
  --max_steps 50 \
  --queue_length 32 \
  --lr 3e-6 \
  --lr_schedule cosine \
  --negative_set paper-count \
  --dropout "$dropout" \
  --reverse_icl_weight 0.0 \
  --momentum_init copy-online \
  --pair_mining csls \
  --pair_csls_k 1 \
  --run_name "smoke_lr_refinement_${wave5_tag}" \
  > "out/zh-lr-refinement/${wave5_tag}_smoke.log" 2>&1

GPU_IDS=0,1,2,3 CONTROL_TAG="$wave5_tag" BASE_DROPOUT="$dropout" \
USE_FNMASK="$use_fnmask" bash scripts/launch_zh_lr_refinement.sh
