#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: $0 /path/to/fine-tuned-llama2-7b-chat [output_dir]}"
OUTPUT_DIR="${2:-./outputs/llama2-7b-agt-ao}"

# Literal Eq.(3) AO uses second-order autodiff, so this reference runner uses
# one GPU. See run_tofu_4gpu_first_order.sh for a practical 4-GPU variant.
CUDA_VISIBLE_DEVICES=0 python train_agt_ao.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name locuslab/TOFU \
  --forget_split forget10 \
  --retain_split retain90 \
  --lr 1e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_epochs 5 \
  --ao_gamma 1.0 \
  --rho 0.6 \
  --warmup_steps auto \
  --tau_grad auto \
  --perturb_layer 10 \
  --inner_loop_steps 4 \
  --adv_epsilon 1e-2 \
  --adv_alpha 5e-3 \
  --beta 0.1 \
  --ao_impl exact \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05
