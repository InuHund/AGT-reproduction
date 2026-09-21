from __future__ import annotations

import contextlib
import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    return local_rank, rank, world


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def global_grad_norm(model) -> float:
    sq = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    for p in model.parameters():
        if p.grad is not None:
            sq = sq + p.grad.detach().float().pow(2).sum()
    if dist.is_initialized():
        dist.all_reduce(sq, op=dist.ReduceOp.SUM)
    return float(torch.sqrt(sq.clamp_min(0)).item())


@contextlib.contextmanager
def higher_order_param_context(engine, params, enabled: bool = True):
    """
    Make full ZeRO-3 parameters visible while constructing AO's higher-order graph.

    Exact AO needs g_f(theta) and g_r(theta) to remain differentiable with respect to theta.
    The ordinary ZeRO-3 training state remains partitioned outside this window.
    """
    if not enabled:
        yield
        return

    import deepspeed

    gathered = getattr(deepspeed.zero, "GatheredParameters", None)
    if gathered is None:
        raise RuntimeError(
            "DeepSpeed does not expose deepspeed.zero.GatheredParameters; "
            "cannot safely enter the exact ZeRO-3 higher-order AO path."
        )
    with gathered(params, modifier_rank=None):
        yield
