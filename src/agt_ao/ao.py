from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _safe_grad(g: torch.Tensor | None, p: torch.Tensor) -> torch.Tensor:
    if g is None:
        return torch.zeros_like(p)
    return g


def gradient_geometry(
    grads_f: Sequence[torch.Tensor | None],
    grads_r: Sequence[torch.Tensor | None],
    params: Sequence[torch.Tensor],
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dot product, cosine, and conflict mask while preserving autograd graph."""
    dot = None
    sq_f = None
    sq_r = None
    for gf, gr, p in zip(grads_f, grads_r, params):
        gf = _safe_grad(gf, p)
        gr = _safe_grad(gr, p)
        term_dot = (gf * gr).sum()
        term_f = (gf * gf).sum()
        term_r = (gr * gr).sum()
        dot = term_dot if dot is None else dot + term_dot
        sq_f = term_f if sq_f is None else sq_f + term_f
        sq_r = term_r if sq_r is None else sq_r + term_r

    assert dot is not None and sq_f is not None and sq_r is not None
    denom = torch.sqrt(sq_f.clamp_min(eps) * sq_r.clamp_min(eps))
    cosine = dot / denom
    conflict = dot < 0
    return dot, cosine.clamp(-1.0, 1.0), conflict


def ao_penalty(
    grads_f: Sequence[torch.Tensor | None],
    grads_r: Sequence[torch.Tensor | None],
    params: Sequence[torch.Tensor],
    gamma: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper Eq. (3): AO with a live higher-order graph through both gradient vectors."""
    dot, cosine, conflict = gradient_geometry(grads_f, grads_r, params, eps=eps)
    core = ((1.0 - cosine) / 2.0).pow(gamma)
    penalty = core * conflict.to(core.dtype)
    return penalty, cosine, dot


def exact_objective(
    forget_loss: torch.Tensor,
    retain_loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    gamma: float,
    lambda_ao: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...], tuple[torch.Tensor | None, ...]]:
    grads_f = torch.autograd.grad(
        forget_loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    grads_r = torch.autograd.grad(
        retain_loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    penalty, cosine, dot = ao_penalty(grads_f, grads_r, params, gamma=gamma)
    total = forget_loss + retain_loss + lambda_ao * penalty
    return total, penalty, cosine, grads_f, grads_r


def first_order_projected_objective(
    forget_loss: torch.Tensor,
    retain_loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    gamma: float = 1.0,
    lambda_ao: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnostic first-order variant; AO gradients are detached by design."""
    gf = torch.autograd.grad(forget_loss, params, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(retain_loss, params, retain_graph=True, allow_unused=True)
    gf_det = tuple(None if g is None else g.detach() for g in gf)
    gr_det = tuple(None if g is None else g.detach() for g in gr)
    penalty, cosine, _ = ao_penalty(gf_det, gr_det, params, gamma=gamma)
    total = forget_loss + retain_loss + lambda_ao * penalty.detach()
    return total, penalty.detach(), cosine.detach()
