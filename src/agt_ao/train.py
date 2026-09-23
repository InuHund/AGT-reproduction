from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

from .ao import exact_objective, first_order_projected_objective
from .data import build_tofu_loaders, PairedLoader
from .distributed import init_distributed, is_main, barrier, global_grad_norm, higher_order_param_context
from .losses import simnpo_loss, causal_lm_loss
from .model import LatentAddHook, load_causal_lm


def parse_args():
    p = argparse.ArgumentParser(description="Exact higher-order AGT^AO trainer")
    p.add_argument("--config", default=None)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="outputs/agt_ao_exact")
    p.add_argument("--dataset_name", default="locuslab/TOFU")
    p.add_argument("--forget_split", default="forget10")
    p.add_argument("--retain_split", default="retain90")
    p.add_argument("--backend", choices=["zero3", "fsdp"], default="zero3")
    p.add_argument("--ao_mode", choices=["exact", "first_order", "none"], default="exact")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--ao_gamma", type=float, default=1.0)
    p.add_argument("--lambda_ao", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--perturb_layer", type=int, default=10)
    p.add_argument("--inner_steps", type=int, default=4)
    p.add_argument("--adv_epsilon", type=float, default=1e-2)
    p.add_argument("--adv_alpha", type=float, default=5e-3)
    p.add_argument("--simnpo_beta", type=float, default=0.1)
    p.add_argument("--simnpo_margin", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=float("inf"))
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=-1, help="Maximum optimizer updates; -1 means full training")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--wandb_project", default="agt-ao-repro")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="disabled")
    p.add_argument("--zero3_gather_for_higher_order", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_config(args):
    if not args.config:
        return args
    data = yaml.safe_load(Path(args.config).read_text()) or {}
    aliases = {
        "learning_rate": "lr",
        "batch_size_per_gpu": "batch_size",
    }
    for key, value in data.items():
        key = aliases.get(key, key)
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_init_wandb(args):
    if args.wandb_mode == "disabled" or not is_main():
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )


def build_zero3_config(args):
    path = Path(__file__).resolve().parents[2] / "configs" / "zero3_bf16.json"
    cfg = json.loads(path.read_text())
    cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    return cfg


def save_state(engine, output_dir, step):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # DeepSpeed checkpointing is a collective operation: every rank must enter it.
    engine.save_checkpoint(str(out), tag=f"checkpoint-{step:05d}")


def compute_total(args, model, fb, rb, delta, params):
    with LatentAddHook(model, args.perturb_layer, delta) as hook:
        out_f = model(**fb)
        actual_delta = hook.delta
    out_r = model(**rb)
    lf = simnpo_loss(
        out_f.logits,
        fb["labels"],
        beta=args.simnpo_beta,
        margin=args.simnpo_margin,
    )
    lr = causal_lm_loss(out_r.logits, rb["labels"])
    if args.ao_mode == "none":
        total = lf + lr
        penalty = torch.zeros((), device=lf.device)
        cosine = torch.zeros((), device=lf.device)
    elif args.ao_mode == "exact":
        total, penalty, cosine, _, _ = exact_objective(
            lf, lr, params, args.ao_gamma, args.lambda_ao
        )
    else:
        total, penalty, cosine = first_order_projected_objective(
            lf, lr, params, args.ao_gamma, args.lambda_ao
        )
    return total, lf, lr, penalty, cosine, actual_delta


def clean_gradient_norm(total, params, device):
    """Compute ||grad_theta L||_2 while allowing the current computation graph to be freed."""
    grads = torch.autograd.grad(
        total,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    sq = torch.zeros((), device=device, dtype=torch.float32)
    for g in grads:
        if g is not None:
            sq = sq + g.detach().float().pow(2).sum()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sq, op=torch.distributed.ReduceOp.SUM)
    return float(torch.sqrt(sq.clamp_min(0)).item())


def pgd_attack(args, model, fb, rb, delta, params):
    """Eq. (6): L_inf PGD on the latent perturbation, using the full Eq. (4) objective."""
    cur = delta.detach().clone()
    for _ in range(args.inner_steps):
        cur.requires_grad_(True)
        with LatentAddHook(model, args.perturb_layer, cur) as hook:
            out_f = model(**fb)
        out_r = model(**rb)
        lf = simnpo_loss(
            out_f.logits,
            fb["labels"],
            beta=args.simnpo_beta,
            margin=args.simnpo_margin,
        )
        lr = causal_lm_loss(out_r.logits, rb["labels"])
        if args.ao_mode == "exact":
            attack_loss, penalty, cosine, _, _ = exact_objective(
                lf, lr, params, args.ao_gamma, args.lambda_ao
            )
        else:
            attack_loss, penalty, cosine = first_order_projected_objective(
                lf, lr, params, args.ao_gamma, args.lambda_ao
            )

        g_delta = torch.autograd.grad(
            attack_loss,
            cur,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        with torch.no_grad():
            cur = (cur + args.adv_alpha * g_delta.sign()).clamp(
                -args.adv_epsilon,
                args.adv_epsilon,
            )
        del attack_loss, penalty, cosine, lf, lr, out_f, out_r
    return cur.detach()


def main():
    args = load_config(parse_args())
    local_rank, rank, world = init_distributed()
    set_seed(args.seed + rank)

    if args.backend == "fsdp" and args.ao_mode == "exact":
        raise RuntimeError(
            "Exact higher-order AO is blocked under current PyTorch FSDP because FSDP does not support double backward. "
            "Use backend=zero3 for the exact path; FSDP first_order is retained as a diagnostic path."
        )

    device = torch.device(
        "cuda", local_rank if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    loaded = load_causal_lm(
        args.model_path,
        dtype=dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = loaded.model.to(device)
    tokenizer = loaded.tokenizer
    for p in model.parameters():
        p.requires_grad_(True)

    _, _, forget_loader, retain_loader, forget_sampler, retain_sampler = build_tofu_loaders(
        tokenizer,
        args.dataset_name,
        args.forget_split,
        args.retain_split,
        args.batch_size,
        args.max_length,
        rank,
        world,
        args.seed,
    )
    paired = PairedLoader(forget_loader, retain_loader)

    if args.backend == "zero3":
        import deepspeed

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        ds_cfg = build_zero3_config(args)
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=params,
            optimizer=optimizer,
            config=ds_cfg,
        )
        model = engine
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            device_id=device,
            use_orig_params=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        engine = None

    params = [p for p in model.parameters() if p.requires_grad]
    wb = maybe_init_wandb(args)

    steps_per_epoch = max(len(forget_loader), len(retain_loader))
    persistent_delta = None
    tau_grad = None
    global_step = 0
    micro_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        forget_sampler.set_epoch(epoch)
        retain_sampler.set_epoch(epoch)
        epoch_started = time.time()

        for fb, rb in paired:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            micro_step += 1
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            fb = {k: v.to(device, non_blocking=True) for k, v in fb.items()}
            rb = {k: v.to(device, non_blocking=True) for k, v in rb.items()}
            hidden_size = (
                model.module.config.hidden_size
                if hasattr(model, "module")
                else model.config.hidden_size
            )
            if persistent_delta is None or persistent_delta.shape != (
                fb["input_ids"].size(0),
                fb["input_ids"].size(1),
                hidden_size,
            ):
                persistent_delta = torch.zeros(
                    fb["input_ids"].size(0),
                    fb["input_ids"].size(1),
                    hidden_size,
                    device=device,
                    dtype=dtype,
                )

            context = higher_order_param_context(
                model,
                params,
                enabled=(
                    args.backend == "zero3"
                    and args.ao_mode == "exact"
                    and args.zero3_gather_for_higher_order
                ),
            )
            with context:
                if args.ao_mode == "none":
                    clean_norm = float("nan")
                    at_warmup_end = False
                    active = False
                else:
                    clean_total, clean_lf, clean_lr, clean_penalty, clean_cos, _ = compute_total(
                        args, model, fb, rb, persistent_delta.detach(), params
                    )
                    # Consume the clean graph before constructing the PGD graphs.
                    clean_norm = clean_gradient_norm(clean_total, params, device)

                    # The paper disables adversarial training for an initial warm-up epoch and
                    # defines tau_grad from the final warm-up loss gradient norm.
                    at_warmup_end = epoch == args.warmup_epochs - 1 and (
                        micro_step % steps_per_epoch == 0
                    )
                    if args.warmup_epochs == 0 and tau_grad is None:
                        tau_grad = args.rho * clean_norm
                        at_warmup_end = False
                    if at_warmup_end:
                        tau_grad = args.rho * clean_norm
                        if is_main():
                            print(
                                f"[GBG] warmup finished: clean_grad_norm={clean_norm:.6g}, "
                                f"tau_grad={tau_grad:.6g}"
                            )

                    active = (
                        epoch >= args.warmup_epochs
                        and tau_grad is not None
                        and clean_norm < tau_grad
                        )
                # clean_total is no longer backpropagatable because clean_gradient_norm consumed it.
                # Recompute the outer objective from scratch, after the gate/inner loop decision.
                if active:
                    persistent_delta = pgd_attack(
                        args, model, fb, rb, persistent_delta.detach(), params
                    )
                outer_total, lf, lr, penalty, cosine, actual_delta = compute_total(
                    args, model, fb, rb, persistent_delta.detach(), params
                )

                if args.backend == "zero3":
                    assert engine is not None
                    engine.backward(outer_total)
                    if not math.isinf(args.max_grad_norm):
                        engine.clip_grad_norm(args.max_grad_norm)
                    grad_norm = global_grad_norm(engine)
                    boundary = engine.is_gradient_accumulation_boundary()
                    engine.step()
                    if boundary:
                        global_step += 1
                else:
                    outer_total.backward()
                    if not math.isinf(args.max_grad_norm):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    grad_norm = global_grad_norm(model)
                    if micro_step % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

            if is_main() and (micro_step % args.log_every == 0):
                peak_mem = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if torch.cuda.is_available()
                    else 0.0
                )
                updates_per_sec = global_step / max(time.time() - start_time, 1e-6)
                metrics = {
                    "train/forget_loss": float(lf.detach().item()),
                    "train/retain_loss": float(lr.detach().item()),
                    "train/ao_penalty": float(penalty.detach().item()),
                    "train/ao_cosine": float(cosine.detach().item()),
                    "train/clean_grad_norm": clean_norm,
                    "train/grad_norm_after_backward": grad_norm,
                    "train/gbg_active": int(active),
                    "train/tau_grad": float(tau_grad) if tau_grad is not None else float("nan"),
                    "train/delta_linf": float(actual_delta.detach().abs().max().item()) if actual_delta is not None else 0.0,
                    "train/micro_step": micro_step,
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                    "system/gpu_mem_gb": peak_mem,
                    "system/updates_per_sec": updates_per_sec,
                    "system/epoch_elapsed_s": time.time() - epoch_started,
                    "system/elapsed_s": time.time() - start_time,
                }
                print(metrics)
                if wb is not None:
                    wb.log(metrics, step=micro_step)

            if (
                args.save_every > 0
                and global_step > 0
                and global_step % args.save_every == 0
                and args.backend == "zero3"
            ):
                if is_main():
                    print(f"Saving ZeRO-3 checkpoint at optimizer step {global_step}")
                save_state(engine, args.output_dir, global_step)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    barrier()
    if args.backend == "zero3":
        save_state(engine, args.output_dir, global_step)
    if wb is not None:
        wb.finish()
    if is_main():
        print(
            f"Finished. output_dir={args.output_dir}, optimizer_steps={global_step}, "
            f"tau_grad={tau_grad}"
        )


if __name__ == "__main__":
    main()
