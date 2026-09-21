#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${1:?usage: $0 /path/to/model}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node=4 \
  -m agt_ao.train \
  --model_path "$MODEL_PATH" \
  --output_dir outputs/smoke_agt_ao \
  --backend zero3 \
  --ao_mode exact \
  --dataset_name locuslab/TOFU \
  --forget_split forget10 \
  --retain_split retain90 \
  --lr 1e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 2 \
  --epochs 1 \
  --ao_gamma 1 \
  --lambda_ao 1 \
  --warmup_epochs 0 \
  --rho 0.6 \
  --perturb_layer 10 \
  --inner_steps 1 \
  --adv_epsilon 1e-2 \
  --adv_alpha 5e-3 \
  --simnpo_beta 0.1 \
  --simnpo_margin 0.0 \
  --max_length 256 \
  --max_steps 2
