#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${1:?usage: $0 /path/to/hf/model [reference_model]}"
REFERENCE_PATH="${2:-}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

ARGS=(
  --model_path "$MODEL_PATH"
  --dataset_name locuslab/TOFU
  --forget_split forget10
  --retain_split retain90
  --max_eval_samples 200
  --max_new_tokens 64
  --output_json "outputs/eval_tofu.json"
)
if [[ -n "$REFERENCE_PATH" ]]; then
  ARGS+=(--reference_model_path "$REFERENCE_PATH")
fi

python -m agt_ao.evaluate "${ARGS[@]}"
