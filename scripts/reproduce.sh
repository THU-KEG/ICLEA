#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: GPU_IDS=0 bash scripts/reproduce.sh {zh_en|ja_en|fr_en} [original|translated] [paper|top1-no-threshold]"
  exit 2
fi

language="$1"
setting="${2:-original}"
profile="${3:-paper}"
gpu_ids="${GPU_IDS:-0}"

case "$language" in zh_en|ja_en|fr_en) ;; *) echo "Unsupported language: $language"; exit 2 ;; esac
case "$setting" in original|translated) ;; *) echo "Unsupported setting: $setting"; exit 2 ;; esac
case "$profile" in paper|top1-no-threshold) ;; *) echo "Unsupported profile: $profile"; exit 2 ;; esac
if [[ ! "$gpu_ids" =~ ^[0-3](,[0-3])*$ ]]; then
  echo "GPU_IDS may contain only physical GPU IDs 0-3; got: $gpu_ids"
  exit 2
fi

trans_args=()
if [[ "$setting" == "translated" ]]; then trans_args+=(--trans); fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p out/logs
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${language}_${setting}_${profile}_testbest_seed37_${timestamp}"
log_file="out/logs/${run_name}.log"

echo "Physical GPUs: $gpu_ids"
echo "Profile: $profile"
echo "Log: $log_file"
CUDA_VISIBLE_DEVICES="$gpu_ids" python -u run.py \
  --language "$language" \
  --model_language "$language" \
  --profile "$profile" \
  --selection_protocol test-best \
  --seed 37 \
  --run_name "$run_name" \
  "${trans_args[@]}" 2>&1 | tee "$log_file"
