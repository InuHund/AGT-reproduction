from __future__ import annotations

import argparse
import torch
from .ao import exact_objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    d = torch.device(a.device)
    theta = torch.tensor([0.0, 1.0], device=d, requires_grad=True)
    # Construct two conflicting objectives whose gradients remain theta-dependent.
    lf = (theta[0] - 1.0).pow(2) + 0.2 * theta[1].pow(2)
    lr = (theta[0] + 1.0).pow(2) + 0.3 * theta[1].pow(2)
    total, penalty, cosine, gf, gr = exact_objective(lf, lr, [theta], gamma=1.0, lambda_ao=1.0)
    g = torch.autograd.grad(total, theta)[0]
    print({
        "forget_loss": float(lf.detach()),
        "retain_loss": float(lr.detach()),
        "ao_penalty": float(penalty.detach()),
        "cosine": float(cosine.detach()),
        "ao_requires_grad": bool(penalty.requires_grad),
        "theta_update_grad": g.detach().cpu().tolist(),
    })
    assert penalty.requires_grad
    assert torch.isfinite(g).all()
    assert g.abs().sum() > 0


if __name__ == "__main__":
    main()
