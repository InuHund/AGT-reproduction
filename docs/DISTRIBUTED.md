# Distributed execution notes

## Exact path: DeepSpeed ZeRO-3

`AGT^AO` in this repository is the higher-order interpretation of the paper objective:

1. Compute the forget gradient `g_f = grad(L_forget, theta, create_graph=True)`.
2. Compute the retain gradient `g_r = grad(L_retain, theta, create_graph=True)`.
3. Form the live AO scalar from those two gradient vectors.
4. Differentiate the resulting `L_unlearn` again with respect to `theta` for the optimizer update.

This creates a genuine gradient-of-gradient workload. ZeRO-3 remains the main distributed backend because it partitions parameters, gradients, and optimizer states during ordinary training. Inside the exact AO region, parameters are gathered with `GatheredParameters` so the higher-order graph is built over complete logical parameters; after leaving the context, the ordinary ZeRO-3 state is partitioned again.

This is deliberately conservative. It should not be read as a claim that arbitrary higher-order autograd is natively transparent through every ZeRO-3 operation. Start with `scripts/smoke_zero3_4gpu.sh` and treat a successful short run as a prerequisite for the long experiment.

## FSDP

Current PyTorch FSDP documentation lists double backward as a limitation. Therefore the exact path fails fast under `--backend fsdp --ao_mode exact`. The repository includes `fsdp + first_order` only as a sharded diagnostic/baseline path.

## Memory strategy

The clean gate gradient is computed with `retain_graph=False`. The clean graph is therefore freed before the adversarial PGD inner loop begins. The outer objective is recomputed afterward. This costs an extra forward pass but prevents the clean higher-order graph from sitting in memory simultaneously with the PGD graphs.

For a 7B model, keep:

- BF16 parameters
- gradient checkpointing enabled
- per-GPU batch size 1
- gradient accumulation 8
- sequence length 512 initially

Then increase sequence length/batch only after checking peak memory.
