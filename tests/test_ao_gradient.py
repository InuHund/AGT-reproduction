import torch

from agt_ao.ao import exact_objective


def test_ao_has_live_parameter_graph_and_gradient():
    theta = torch.tensor([0.0, 1.0], requires_grad=True)
    lf = (theta[0] - 1.0).pow(2) + 0.2 * theta[1].pow(2)
    lr = (theta[0] + 1.0).pow(2) + 0.3 * theta[1].pow(2)
    total, penalty, cosine, *_ = exact_objective(lf, lr, [theta], gamma=1.0, lambda_ao=1.0)
    assert penalty.requires_grad
    assert penalty.grad_fn is not None
    assert float(cosine.detach()) < 0.0
    grad_ao = torch.autograd.grad(penalty, theta, retain_graph=True)[0]
    assert torch.isfinite(grad_ao).all()
    assert grad_ao.abs().sum() > 0
    grad_total = torch.autograd.grad(total, theta)[0]
    assert torch.isfinite(grad_total).all()
