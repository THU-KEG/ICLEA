#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${ICLEA_PYTHON:-python}"
tag="${DESCRIPTION_TAG:-20260830_hits10_wave5_description_scale}"
log_root="out/zh-hits10-description-scale"
run_root="${log_root}/${tag}"
audit_root="out/zh-hits10/${tag}"
diagnostic_root="out/diagnostics"
mkdir -p "$audit_root" "$diagnostic_root"

while [[ ! -s "${run_root}/manifest.tsv" ]]; do
  sleep 60
done

while [[ "$(find out/checkpoints -path "*${tag}/snapshot_epoch_151.pt" | wc -l)" -lt 4 ]]; do
  sleep 60
done

"$python_bin" scripts/status_hits10_wave.py \
  --tag "$tag" \
  --root "$log_root" \
  --report-epoch 151 \
  --output "${audit_root}/trajectory_epoch151.json"

nearfreeze_count="$(
  find out/checkpoints -path "*nearfreeze*${tag}/snapshot_epoch_151.pt" | wc -l
)"
if [[ "$nearfreeze_count" -ne 3 ]]; then
  echo "expected exactly three matched near-freeze epoch-151 checkpoints, found ${nearfreeze_count}" >&2
  exit 2
fi
mapfile -t nearfreeze_checkpoints < <(
  find out/checkpoints -path "*nearfreeze*${tag}/snapshot_epoch_151.pt" | sort
)

# The SSH account inherits a negative nice value on this server. Set the
# monitor shell to absolute nice=19 so the CPU evaluator child inherits it.
renice 19 -p "$$" >/dev/null
for nearfreeze_checkpoint in "${nearfreeze_checkpoints[@]}"; do
  run_name="$(basename "$(dirname "$nearfreeze_checkpoint")")"
  before_checkpoint="$(dirname "$nearfreeze_checkpoint")/snapshot_epoch_150.pt"
  CUDA_VISIBLE_DEVICES="" \
  taskset -c 80-83 \
    "$python_bin" scripts/audit_checkpoint_drift.py \
      --before "$before_checkpoint" \
      --after "$nearfreeze_checkpoint" \
      --weight-source online \
      --maximum-absolute-delta 1e-10 \
      --require-before-epoch 150 \
      --require-after-epoch 151 \
      --output "${diagnostic_root}/${run_name}_epoch150_to_151_online_drift.json"
  CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  taskset -c 80-83 \
    "$python_bin" -u scripts/evaluate_checkpoints.py \
      --checkpoint "$nearfreeze_checkpoint" \
      --candidate-scope full-target \
      --weight-source online \
      --device cpu \
      --output "${diagnostic_root}/${run_name}_epoch151_online_full_target.json" \
      > "${audit_root}/${run_name}_epoch151_independent_cpu.log" 2>&1
done

echo "description_epoch151_audit_complete tag=${tag} checkpoints=${nearfreeze_checkpoints[*]}"
