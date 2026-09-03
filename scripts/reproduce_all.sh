#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
gpu_ids="${GPU_IDS:-0,1,2,3}"
profile="${PROFILE:-paper}"
if [[ ! "$gpu_ids" =~ ^[0-3](,[0-3])*$ ]]; then
  echo "GPU_IDS may contain only physical GPU IDs 0-3; got: $gpu_ids"
  exit 2
fi

for setting in original translated; do
  for language in zh_en ja_en fr_en; do
    python scripts/run_five.py \
      --language "$language" \
      --setting "$setting" \
      --profile "$profile" \
      --gpus "$gpu_ids"
  done
done
