from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log p(y_t | x, y_<t) over non-masked target tokens, per example."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clamp_min(0)
    token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = shift_labels.ne(-100)
    return (token_lp * mask).sum(dim=-1)


def simnpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    margin: float = 0.0,
) -> torch.Tensor:
    """Paper Eq. (2) / appendix SimNPO-style reference-free forget loss."""
    logp = sequence_logprob(logits, labels)
    lengths = labels[:, 1:].ne(-100).sum(dim=-1).clamp_min(1)
    normalized_logp = logp / lengths
    z = -beta * normalized_logp - margin
    return (-2.0 / beta * F.logsigmoid(z)).mean()


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
