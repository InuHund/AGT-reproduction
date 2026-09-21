# Standalone AGT^AO reproduction

This is a clean standalone implementation of **AGT^AO** for causal LMs, with TOFU `forget10/retain90` as the first target. It is intentionally separate from the released repository so the reproduction code is easy to inspect and modify.

Paper: Li et al., *AGT^AO: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality*, arXiv:2602.01703.

## What is fixed by the paper

For the Llama-2-7B-chat TOFU setting, the paper reports:

- unlearning LR: `1e-4`
- batch size: `1`
- gradient accumulation: `8`
- unlearning epochs: `5`
- AO `gamma`: `1`
- GBG warmup: first epoch
- `rho = 0.6`
- perturbation layer: `10` for 7B
- inner attack steps: `4`

The paper says that the gradient threshold is `tau_grad = rho * ||grad at the final warmup step||_2` and that the attack is activated only when the current unlearning gradient norm is below that threshold.

The released repository also supplies numerical defaults for quantities that are not specified in Appendix A.3: `beta=0.1`, `adv_epsilon=1e-2`, and `adv_alpha=5e-3`. This implementation exposes all three as CLI arguments rather than hiding them.

## Important reproducibility caveats

There are a few mismatches between the paper and the released implementation:

1. The paper's Eq. (3)/(4) introduces an AO coefficient `lambda_ao`, but Appendix A.3 does not give a numerical value. This script exposes `--ao_lambda` and defaults it to `1.0`.
2. The paper defines AO as a function of parameter gradients. A mathematically literal implementation therefore requires higher-order differentiation (`create_graph=True`). The current released code accumulates detached gradients and adds the resulting penalty as a scalar, which makes that penalty disconnected from model parameters. This script's default `--ao_impl exact` implements the differentiable form of the paper equation.
3. The released code performs the PGD attack on NPO + retain loss without including AO in the inner objective. The default here is `--attack_include_ao` off for compatibility with that released code. The flag is available for experiments with an inner objective that also contains AO.
4. The public README command uses fixed `--warmup_steps 50` and `--tau_grad 2.0`, whereas Appendix A.3 specifies an adaptive one-epoch warmup with `rho=0.6`. The default here follows Appendix A.3, not the README shortcut.
5. Literal AO depends on gradients with respect to model parameters, so differentiating it requires second-order autodiff. The `exact` runner therefore uses one GPU; the included 4-GPU runner uses the first-order conflict-projection approximation instead.

Because of these differences, an exact numerical match to Table 1 cannot be claimed from the paper alone.

## Install

```bash
pip install torch transformers datasets accelerate peft
```

For the user's Ubuntu/A100 setup, use a PyTorch build matching the installed CUDA driver/runtime.

## Run

For the paper-style TOFU configuration with literal differentiable AO:

```bash
CUDA_VISIBLE_DEVICES=0 python train_agt_ao.py \
  --model_path /path/to/your/fine_tuned_llama2_7b_chat \
  --output_dir ./outputs/llama2-7b_agt_ao \
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
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

The input model is the **fine-tuned target model to be unlearned**. The reference model is loaded from the same checkpoint before the unlearning updates, matching the released implementation.

For a very small smoke test, add:

```bash
--max_train_steps 2 --max_length 128
```

## Full fine-tuning

Set `--lora_r 0`. This is much more memory-intensive because the full 7B model is trainable.

## Why the perturbation is persistent

The released implementation keeps an `adversarial_delta` tensor on the trainer and warm-starts the next PGD search from the previous value. This script follows that behavior after the GBG gate turns on.

## 4-GPU practical run

For the A100 x4 setup, use `run_tofu_4gpu_first_order.sh`; it keeps the paper hyperparameters but uses the first-order AO projection to avoid the second-order distributed-autodiff issue.
