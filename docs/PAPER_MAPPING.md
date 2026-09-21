# Paper-to-code mapping

| Paper | Code |
|---|---|
| Eq. (2), SimNPO-style forget loss | `src/agt_ao/losses.py::simnpo_loss` |
| Eq. (3), AO | `src/agt_ao/ao.py::ao_penalty` |
| Eq. (4), unified objective | `src/agt_ao/ao.py::exact_objective` |
| Eq. (5), latent min-max | `src/agt_ao/train.py::pgd_attack` + outer objective |
| Eq. (6), L_inf PGD | `src/agt_ao/train.py::pgd_attack` |
| Eq. (7), outer parameter update | `src/agt_ao/train.py` after the inner loop |
| GBG warm-up / threshold | `src/agt_ao/train.py` |
| 7B perturbation layer 10 | CLI default/config |
| inner steps 4 | CLI default/config |

## Higher-order requirement

`g_f` and `g_r` are produced with `create_graph=True`. The AO scalar therefore has a live dependency on the model parameters. The final backward differentiates through those gradient computations.

Do **not** add `.detach()` to `g_f` or `g_r` in the exact implementation, do not turn them into CPU tensors, and do not construct AO under `torch.no_grad()`.

## Gate graph lifetime

The gate needs `||grad L_unlearn||_2`. We first construct the exact objective and consume its graph with `autograd.grad(..., retain_graph=False)`. If the attack is enabled, the PGD loop then builds fresh graphs; the final outer objective is recomputed from scratch. This is intentional peak-memory control.

## Paper vs completion values

The paper explicitly reports, for the 7B unlearning setup: LR `1e-4`, batch size `1`, gradient accumulation `8`, 5 epochs, AdamW, AO gamma `1`, one warm-up epoch, `rho=0.6`, perturb layer `10`, inner steps `4`. The paper's Appendix A.3 does not specify all remaining implementation-level values. This repo exposes those values as CLI arguments and records the completion values in the run config rather than presenting them as paper-reported numbers.

## Repository discrepancy

The released repository is used only as a cross-check. In the exact implementation here, stored AO gradients are intentionally **not detached** because detaching would sever the dependency of `R_AO` on `theta` and make its parameter-update contribution zero in ordinary autograd.
