# AGT^AO exact higher-order reproduction

This repository is a clean-room reproduction of the mathematical training objective described in:

> Pengyu Li et al., **AGT^AO: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality**, arXiv:2602.01703 (2026).

Paper: https://arxiv.org/abs/2602.01703
Official code (used only for implementation cross-checks): https://github.com/TiezMind/AGT-unlearning

## What is different from the released implementation?

The paper defines

`R_AO = I(g_f · g_r < 0) * ((1 - cos(g_f, g_r)) / 2)^gamma`

with `g_f = grad_theta(L_forget)` and `g_r = grad_theta(L_retain)`, and uses

`L_unlearn = L_forget + L_retain + lambda_ao * R_AO`.

For AO to affect a parameter update, the implementation below deliberately keeps `g_f` and `g_r` connected to `theta` with `create_graph=True`. The outer backward therefore differentiates through the gradient computation (a higher-order / gradient-of-gradient computation).

The released AGT.py currently detaches stored gradients before building its penalty and later adds the penalty as a constant tensor. That makes the displayed penalty unable to contribute a gradient back to model parameters in the usual computational-graph sense. The repository also uses an NPO/reference-model loss in its AGT path, while the paper's main Eq. (2) is the reference-free SimNPO-style form. This repo follows the paper equations rather than copying those behaviors.

## Important distributed-training constraint

The exact higher-order AO path uses DeepSpeed ZeRO-3 as the primary 4-GPU backend. During the higher-order AO window, trainable parameters are gathered so PyTorch autograd can build the gradient graph over the logical parameter tensors; optimizer state and the ordinary ZeRO-3 model state remain sharded. This is intentionally more conservative than pretending that arbitrary double-backward is transparently supported by a sharded wrapper.

Legacy PyTorch FSDP explicitly documents that it does not support double backward. Therefore `--backend fsdp --ao-mode exact` fails fast rather than claiming to implement the requested mathematics. FSDP first-order mode is included for diagnostics/baselines.

## Paper settings vs completion values

Paper-specified for 7B unlearning:

- learning rate: `1e-4`
- per-device batch size: `1`
- gradient accumulation: `8`
- epochs: `5`
- AO gamma: `1`
- warm-up: `1 epoch`
- GBG rho: `0.6`
- perturbation layer: `10` (7B)
- inner PGD steps: `4`
- optimizer: AdamW

Values not specified in the paper are exposed as arguments and are completed using the prior defaults / released implementation where available:

- SimNPO beta: `0.1`
- SimNPO margin/delta: `0.0`
- latent PGD epsilon: `1e-2`
- latent PGD step alpha: `5e-3`
- AO coefficient lambda_ao: `1.0`
- max grad norm: `inf`

These completion values are **not claimed to be paper-reported hyperparameters**.

## Main run: 4 x A100 80 GB

1. Create the environment:

```bash
conda create -n agt-ao-exact python=3.10 -y
conda activate agt-ao-exact
pip install -r requirements.txt
```

2. Authenticate to Hugging Face if needed for LLaMA 2:

```bash
huggingface-cli login
```

3. Start with a 2-step smoke test:

```bash
bash scripts/smoke_zero3_4gpu.sh /path/to/your/Llama-2-7b-chat-checkpoint
```

4. Run the paper configuration:

```bash
bash scripts/run_zero3_4gpu.sh /path/to/your/Llama-2-7b-chat-checkpoint
```

The default dataset is `locuslab/TOFU`, with `forget10` and `retain90`.

## Checkpoint conversion

DeepSpeed ZeRO-3 checkpoints are sharded. After training, convert a ZeRO checkpoint to a normal Hugging Face model before evaluation:

```bash
bash scripts/zero3_to_hf.sh outputs/agt_ao_exact/checkpoint-00001 outputs/agt_ao_exact/hf_model
```

## Evaluation

The evaluator measures:

- mean token NLL on forget / retain
- mean target-sequence log-probability
- generated answer ROUGE-L on forget / retain
- perplexity
- optional comparison against a target/reference checkpoint

The paper's full KUR/PLR pipeline is broader than a lightweight standalone evaluator. This repository reports the primitive quantities directly and does not invent a KUR implementation where the required benchmark auxiliary data are unavailable.

```bash
python -m agt_ao.evaluate \
  --model_path outputs/agt_ao_exact/hf_model \
  --dataset_name locuslab/TOFU \
  --forget_split forget10 \
  --retain_split retain90 \
  --max_eval_samples 200 \
  --wandb_project agt-ao-repro
```

## AO correctness test

The included unit test checks that the AO term retains a live computational graph and that differentiating it produces a non-zero parameter gradient in a controlled conflicting-gradient example:

```bash
pytest -q
```

## Reproducibility notes

- Seed is explicit (`42` by default).
- All paper and completion hyperparameters are recorded in the run config and W&B config.
- Training logs include clean objective loss, AO penalty, gradient cosine, clean gradient norm, gate activation, delta norm, memory, throughput, and optimizer steps.
- `torch.compile` is intentionally not enabled in the exact AO path because current PyTorch releases have documented issues involving higher-order gradient graph preservation under compilation.

## Validation workflow

Run the mathematical AO test first:

```bash
pytest -q
python -m agt_ao.smoke --device cpu
```

Then run the distributed GPU smoke test:

```bash
bash scripts/smoke_zero3_4gpu.sh /path/to/Llama-2-7b-chat
```

The smoke run is intentionally short and exists to catch model-loading, ZeRO-3, higher-order autograd, latent-hook, and optimizer-step failures before spending a full experiment budget.

For W&B:

```bash
WANDB_MODE=online bash scripts/run_zero3_4gpu.sh /path/to/Llama-2-7b-chat
```

Training records losses, AO cosine/penalty, gate activation, threshold, perturbation norm, peak GPU memory, and update throughput. Evaluation can also upload its JSON-derived metrics with `--wandb_project`.

The standalone evaluator is a diagnostic evaluator rather than a claim of reproducing every detail of the paper's full benchmark harness. In particular, its primitive likelihood/ROUGE metrics are intended to verify before/after behavior; use the benchmark's official metric implementation when making a table directly comparable to the paper.

Compare two evaluation JSON files with:

```bash
python scripts/compare_eval_json.py outputs/before.json outputs/after.json
```
