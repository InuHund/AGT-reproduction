#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: $0 /path/to/fine-tuned-llama2-7b-chat [output_dir]}"
OUTPUT_DIR="${2:-./outputs/llama2-7b-agt-ao-4gpu}"

# Practical 4-GPU runner. AO is realized as a first-order conflict projection,
# avoiding the second-order graph needed by literal Eq.(3).
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_agt_ao.py \
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
  --ao_impl first_order_project \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05
