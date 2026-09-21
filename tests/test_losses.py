import torch
from agt_ao.losses import simnpo_loss


def test_simnpo_finite():
    logits = torch.randn(1, 5, 17, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 4, 6]])
    loss = simnpo_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
