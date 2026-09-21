#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${1:?usage: $0 /path/to/model}"
OUTPUT_DIR="${2:-outputs/agt_ao_fsdp_first_order}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

torchrun --standalone --nproc_per_node=4 \
  -m agt_ao.train \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --backend fsdp \
  --ao_mode first_order \
  --dataset_name locuslab/TOFU \
  --forget_split forget10 \
  --retain_split retain90 \
  --lr 1e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --epochs 5 \
  --ao_gamma 1 \
  --lambda_ao 1 \
  --warmup_epochs 1 \
  --rho 0.6 \
  --perturb_layer 10 \
  --inner_steps 4 \
  --max_length 512
