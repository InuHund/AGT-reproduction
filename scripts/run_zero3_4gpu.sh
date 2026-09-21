#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${1:?usage: $0 /path/to/model [output_dir]}"
OUTPUT_DIR="${2:-outputs/agt_ao_exact}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node=4 \
  -m agt_ao.train \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --backend zero3 \
  --ao_mode exact \
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
  --adv_epsilon 1e-2 \
  --adv_alpha 5e-3 \
  --simnpo_beta 0.1 \
  --simnpo_margin 0.0 \
  --max_length 512 \
  --wandb_project agt-ao-repro \
  --wandb_mode "${WANDB_MODE}" \
  "$@"
