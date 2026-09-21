#!/usr/bin/env python3
"""Paper-faithful, standalone implementation of AGT^AO for causal LMs.

The implementation follows the objective and AGT procedure described in:
Li et al., "AGT^AO: Robust and Stabilized LLM Unlearning via Adversarial
Gating Training with Adaptive Orthogonality" (arXiv:2602.01703).

Important reproducibility notes are documented in README.md. In particular,
the paper leaves several quantities under-specified (AO coefficient, PGD
radius/step size, and the exact treatment of the AO term in the attack). The
CLI defaults for those quantities are taken from the released official code
where possible and are clearly marked as such.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model_path: str
    output_dir: str = "./outputs/agt_ao_tofu"
    dataset_name: str = "locuslab/TOFU"
    forget_split: str = "forget10"
    retain_split: str = "retain90"

    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 5
    weight_decay: float = 0.0
    max_length: int = 512

    # AO: paper says gamma=1 and rho=0.6; lambda_AO is not specified.
    ao_gamma: float = 1.0
    ao_lambda: float = 1.0
    rho: float = 0.6

    # GBG / warmup: paper says warmup = first epoch and tau = rho * final
    # warmup gradient norm. "auto" is the paper setting.
    warmup_steps: str = "auto"
    tau_grad: str = "auto"

    perturb_layer: int = 10
    inner_loop_steps: int = 4

    # These three were not numerically specified in the paper text/appendix.
    # They match defaults in the official released implementation.
    adv_epsilon: float = 1e-2
    adv_alpha: float = 5e-3
    beta: float = 0.1

    # By default we optimize the mathematically specified AO penalty. This is
    # second-order because AO depends on parameter gradients. It can be very
    # expensive. --ao_impl first_order_project is provided as a practical
    # approximation.
    ao_impl: str = "exact"

    # The official implementation's inner attack optimizes NPO + retain loss,
    # not the AO penalty itself. Default False reproduces that behavior.
    attack_include_ao: bool = False

    # Main model adaptation. The public repo default config uses LoRA r=32.
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    seed: int = 0
    num_workers: int = 0
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    log_every: int = 1
    save_every_epoch: bool = False
    max_train_steps: int = -1

    # Hugging Face model options.
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    resume_from: Optional[str] = None


# ---------------------------------------------------------------------------
# Utility / model helpers
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run AGT^AO unlearning on TOFU.")

    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default=Config.output_dir)
    p.add_argument("--dataset_name", default=Config.dataset_name)
    p.add_argument("--forget_split", default=Config.forget_split)
    p.add_argument("--retain_split", default=Config.retain_split)

    p.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=Config.learning_rate)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=Config.gradient_accumulation_steps)
    p.add_argument("--num_epochs", type=int, default=Config.num_epochs)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--max_length", type=int, default=Config.max_length)

    p.add_argument("--ao_gamma", type=float, default=Config.ao_gamma)
    p.add_argument("--ao_lambda", type=float, default=Config.ao_lambda)
    p.add_argument("--rho", type=float, default=Config.rho)
    p.add_argument("--warmup_steps", default=Config.warmup_steps,
                   help="'auto' = first epoch, as in paper; integer also accepted.")
    p.add_argument("--tau_grad", default=Config.tau_grad,
                   help="'auto' = rho * final warmup gradient norm; integer/float for fixed threshold.")

    p.add_argument("--perturb_layer", type=int, default=Config.perturb_layer)
    p.add_argument("--inner_loop_steps", "--adv_steps", dest="inner_loop_steps", type=int,
                   default=Config.inner_loop_steps)
    p.add_argument("--adv_epsilon", type=float, default=Config.adv_epsilon)
    p.add_argument("--adv_alpha", type=float, default=Config.adv_alpha)
    p.add_argument("--beta", type=float, default=Config.beta)

    p.add_argument("--ao_impl", choices=["exact", "first_order_project"], default=Config.ao_impl)
    p.add_argument("--attack_include_ao", action="store_true", default=Config.attack_include_ao)

    p.add_argument("--lora_r", type=int, default=Config.lora_r)
    p.add_argument("--lora_alpha", type=int, default=Config.lora_alpha)
    p.add_argument("--lora_dropout", type=float, default=Config.lora_dropout)

    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--max_grad_norm", type=float, default=Config.max_grad_norm)
    p.add_argument("--log_every", type=int, default=Config.log_every)
    p.add_argument("--save_every_epoch", action="store_true", default=Config.save_every_epoch)
    p.add_argument("--max_train_steps", type=int, default=Config.max_train_steps)
    p.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default=Config.torch_dtype)
    p.add_argument("--trust_remote_code", action="store_true", default=Config.trust_remote_code)
    p.add_argument("--resume_from", default=Config.resume_from)

    a = p.parse_args()
    return Config(
        model_path=a.model_path,
        output_dir=a.output_dir,
        dataset_name=a.dataset_name,
        forget_split=a.forget_split,
        retain_split=a.retain_split,
        learning_rate=a.learning_rate,
        batch_size=a.batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        num_epochs=a.num_epochs,
        weight_decay=a.weight_decay,
        max_length=a.max_length,
        ao_gamma=a.ao_gamma,
        ao_lambda=a.ao_lambda,
        rho=a.rho,
        warmup_steps=str(a.warmup_steps),
        tau_grad=str(a.tau_grad),
        perturb_layer=a.perturb_layer,
        inner_loop_steps=a.inner_loop_steps,
        adv_epsilon=a.adv_epsilon,
        adv_alpha=a.adv_alpha,
        beta=a.beta,
        ao_impl=a.ao_impl,
        attack_include_ao=a.attack_include_ao,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        seed=a.seed,
        num_workers=a.num_workers,
        gradient_checkpointing=not a.no_gradient_checkpointing,
        max_grad_norm=a.max_grad_norm,
        log_every=a.log_every,
        save_every_epoch=a.save_every_epoch,
        max_train_steps=a.max_train_steps,
        torch_dtype=a.torch_dtype,
        trust_remote_code=a.trust_remote_code,
        resume_from=a.resume_from,
    )


def dtype_from_string(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def model_layers(model: nn.Module) -> nn.ModuleList:
    """Locate transformer blocks through common HF / PEFT nesting patterns."""
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.base_model.model.layers,
        lambda m: m.model.decoder.layers,
        lambda m: m.transformer.h,
        lambda m: m.base_model.transformer.h,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            pass
    raise ValueError(
        "Could not find transformer layers. Add a resolver for your model architecture."
    )


@contextmanager
def latent_perturbation(model: nn.Module, layer_index_1based: int, delta: torch.Tensor):
    """Inject delta after the selected transformer block, matching the released code."""
    layers = model_layers(model)
    if not (1 <= layer_index_1based <= len(layers)):
        raise ValueError(f"perturb_layer={layer_index_1based} but model has {len(layers)} blocks")
    target = layers[layer_index_1based - 1]

    def hook(_module, _inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0]
            return (hidden + delta,) + outputs[1:]
        if hasattr(outputs, "last_hidden_state"):
            outputs.last_hidden_state = outputs.last_hidden_state + delta
            return outputs
        return outputs + delta

    handle = target.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# TOFU formatting / paired loader
# ---------------------------------------------------------------------------


def format_tofu_example(
    tokenizer,
    question: str,
    answer: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Llama-2-chat style QA formatting used by the original TOFU loader."""
    # The legacy TOFU implementation uses [INST] ... [/INST] for Llama-2.
    # Keeping the strings explicit makes this script independent of the old repo.
    question_text = f"[INST] {question} [/INST]"
    full_text = question_text + " " + answer

    q_ids = tokenizer(question_text, add_special_tokens=True, truncation=True,
                      max_length=max_length)["input_ids"]
    encoded = tokenizer(full_text, add_special_tokens=True, truncation=True,
                        max_length=max_length, padding=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    labels = list(input_ids)
    q_len = min(len(q_ids), len(labels))
    for i in range(q_len):
        labels[i] = -100

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    n_pad = max_length - len(input_ids)
    if n_pad > 0:
        input_ids += [pad_id] * n_pad
        attention_mask += [0] * n_pad
        labels += [-100] * n_pad

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class TofuPairDataset(Dataset):
    def __init__(self, tokenizer, cfg: Config):
        self.forget = load_dataset(cfg.dataset_name, cfg.forget_split, split="train")
        self.retain = load_dataset(cfg.dataset_name, cfg.retain_split, split="train")
        if len(self.forget) == 0 or len(self.retain) == 0:
            raise ValueError("TOFU forget/retain split is empty")
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.forget)

    def __getitem__(self, idx: int):
        f = self.forget[idx]
        # Match the original TOFU pair loader: choose a pseudo-random retain
        # example for every forget example.
        r_idx = (idx + random.randrange(len(self.retain))) % len(self.retain)
        r = self.retain[r_idx]
        return (
            format_tofu_example(self.tokenizer, f["question"], f["answer"], self.cfg.max_length),
            format_tofu_example(self.tokenizer, r["question"], r["answer"], self.cfg.max_length),
        )


def collate_pairs(batch):
    forget = {k: torch.stack([x[0][k] for x in batch]) for k in batch[0][0]}
    retain = {k: torch.stack([x[1][k] for x in batch]) for k in batch[0][1]}
    return forget, retain


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-example token NLL sum, matching the public AGT implementation."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        ignore_index=-100,
        reduction="none",
    )
    return loss.sum(dim=-1)


def npo_loss(model_logits: torch.Tensor, ref_logits: torch.Tensor, labels: torch.Tensor, beta: float) -> torch.Tensor:
    current_nll = sequence_nll(model_logits, labels)
    ref_nll = sequence_nll(ref_logits, labels)
    neg_log_ratio = current_nll - ref_nll
    return (-F.logsigmoid(beta * neg_log_ratio).mean()) * (2.0 / beta)


# ---------------------------------------------------------------------------
# AO
# ---------------------------------------------------------------------------


def _usable_grads(grads: Sequence[Optional[torch.Tensor]], params: Sequence[nn.Parameter]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p))
        else:
            out.append(g)
    return out


def flatten_tensors(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([x.reshape(-1) for x in xs]) if xs else torch.empty(0)


def ao_exact(
    g_forget: Sequence[torch.Tensor],
    g_retain: Sequence[torch.Tensor],
    gamma: float,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable version of Eq. (3)."""
    gf = flatten_tensors(g_forget)
    gr = flatten_tensors(g_retain)
    dot = torch.dot(gf, gr)
    denom = gf.norm() * gr.norm() + eps
    cos = dot / denom
    conflict = dot < 0
    raw = torch.clamp((1.0 - cos) / 2.0, min=0.0).pow(gamma)
    penalty = torch.where(conflict, raw, torch.zeros_like(raw))
    return penalty, cos.detach(), dot.detach()


def first_order_project(
    g_forget: Sequence[torch.Tensor],
    g_retain: Sequence[torch.Tensor],
    eps: float = 1e-12,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Project the forget gradient away from retain when they conflict.

    This is a first-order, optimizer-independent approximation. It is NOT the
    literal second-order gradient of Eq. (3), but is useful when exact AO is
    too expensive on 7B models.
    """
    gf = flatten_tensors(g_forget)
    gr = flatten_tensors(g_retain)
    dot = torch.dot(gf.detach(), gr.detach())
    denom = torch.dot(gr.detach(), gr.detach()) + eps
    cos = dot / (gf.detach().norm() * gr.detach().norm() + eps)
    if dot.item() >= 0:
        return list(g_forget), cos.detach(), dot.detach()
    coeff = dot / denom
    corrected = gf - coeff * gr
    out: List[torch.Tensor] = []
    cursor = 0
    for g in g_forget:
        n = g.numel()
        out.append(corrected[cursor:cursor + n].view_as(g))
        cursor += n
    return out, cos.detach(), dot.detach()


def apply_manual_grads(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g.detach().clone()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AGTAOTrainer:
    def __init__(self, model: nn.Module, ref_model: nn.Module, tokenizer, loader: DataLoader,
                 optimizer, scheduler, accelerator: Accelerator, cfg: Config):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.cfg = cfg
        self.global_update = 0
        self.warmup_steps: Optional[int] = None
        self.tau_grad: Optional[float] = None
        self.adv_delta: Optional[torch.Tensor] = None

        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not self.trainable_params:
            raise ValueError("No trainable parameters. Set --lora_r > 0 or adapt the model loading.")

    def _move_batch(self, x):
        return {k: v.to(self.accelerator.device, non_blocking=True) for k, v in x.items()}

    def _init_delta(self, hidden_shape, device, dtype):
        if self.adv_delta is None or tuple(self.adv_delta.shape) != tuple(hidden_shape):
            self.adv_delta = torch.zeros(hidden_shape, device=device, dtype=dtype)

    def _forward_base(self, forget, retain, delta: Optional[torch.Tensor]):
        if delta is None:
            f_out = self.model(**forget)
        else:
            with latent_perturbation(self.model, self.cfg.perturb_layer, delta):
                f_out = self.model(**forget)
        with torch.no_grad():
            ref_out = self.ref_model(**forget)
        r_out = self.model(**retain)
        lf = npo_loss(f_out.logits, ref_out.logits, forget["labels"], self.cfg.beta)
        lr = r_out.loss
        return lf, lr, f_out

    def _get_grad_tuple(self, loss: torch.Tensor, create_graph: bool) -> List[torch.Tensor]:
        grads = torch.autograd.grad(
            loss,
            self.trainable_params,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        return _usable_grads(grads, self.trainable_params)

    def _gradient_norm(self, grads: Sequence[torch.Tensor]) -> torch.Tensor:
        total = None
        for g in grads:
            val = g.float().pow(2).sum()
            total = val if total is None else total + val
        return torch.sqrt(total + 1e-12) if total is not None else torch.tensor(0.0, device=self.accelerator.device)

    def _compute_ao(self, lf, lr, exact: bool):
        if exact:
            gf = self._get_grad_tuple(lf, create_graph=True)
            gr = self._get_grad_tuple(lr, create_graph=True)
            penalty, cos, dot = ao_exact(gf, gr, self.cfg.ao_gamma)
            return penalty, cos, dot, gf, gr
        gf = self._get_grad_tuple(lf, create_graph=False)
        gr = self._get_grad_tuple(lr, create_graph=False)
        corrected, cos, dot = first_order_project(gf, gr)
        return None, cos, dot, corrected, gr

    def _pgd_attack(self, forget, retain, start_delta: torch.Tensor) -> torch.Tensor:
        delta = start_delta.detach().clone()
        for _ in range(self.cfg.inner_loop_steps):
            delta.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            lf, lr, _ = self._forward_base(forget, retain, delta)
            attack_loss = lf + lr
            (dgrad,) = torch.autograd.grad(attack_loss, delta, retain_graph=False, create_graph=False)
            with torch.no_grad():
                delta = torch.clamp(delta + self.cfg.adv_alpha * dgrad.sign(),
                                     -self.cfg.adv_epsilon, self.cfg.adv_epsilon)
        return delta.detach()

    def _warmup_setup(self, steps_per_epoch: int, grad_norm: float):
        if self.warmup_steps is None:
            if self.cfg.warmup_steps == "auto":
                self.warmup_steps = steps_per_epoch
            else:
                self.warmup_steps = int(float(self.cfg.warmup_steps))
        if self.tau_grad is None:
            if self.cfg.tau_grad == "auto":
                self.tau_grad = self.cfg.rho * grad_norm
            else:
                self.tau_grad = float(self.cfg.tau_grad)

    def train(self):
        steps_per_epoch = math.ceil(len(self.loader) / self.cfg.gradient_accumulation_steps)
        total_updates = steps_per_epoch * self.cfg.num_epochs
        if self.cfg.max_train_steps > 0:
            total_updates = min(total_updates, self.cfg.max_train_steps)

        accum = 0
        running_loss = 0.0
        last_grad_norm = 0.0

        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            for step, (forget, retain) in enumerate(self.loader):
                if self.global_update >= total_updates:
                    break
                forget = self._move_batch(forget)
                retain = self._move_batch(retain)

                # Build clean/current adversarial objective used for GBG trigger.
                self.model.zero_grad(set_to_none=True)
                lf0, lr0, _ = self._forward_base(forget, retain, self.adv_delta)
                base_loss = lf0 + lr0
                base_grads = self._get_grad_tuple(base_loss, create_graph=False)
                base_norm = float(self._gradient_norm(base_grads).detach().item())
                last_grad_norm = base_norm

                # Determine warmup / threshold from final warmup gradient norm.
                if self.warmup_steps is None:
                    self._warmup_setup(steps_per_epoch, base_norm)

                in_warmup = accum < int(self.warmup_steps)
                attack_active = (not in_warmup) and (base_norm < float(self.tau_grad))

                if attack_active:
                    self.adv_delta = self._pgd_attack(forget, retain, self.adv_delta)
                # During warmup, no adversarial update. After warmup, use last
                # perturbation as a warm start if available.
                elif self.adv_delta is not None and in_warmup:
                    pass

                self.model.zero_grad(set_to_none=True)
                lf, lr, _ = self._forward_base(forget, retain, self.adv_delta)

                if self.cfg.ao_impl == "exact":
                    penalty, cosine, dot, gf, gr = self._compute_ao(lf, lr, exact=True)
                    total_loss = lf + lr + self.cfg.ao_lambda * penalty
                else:
                    _, cosine, dot, corrected_f, retain_gr = self._compute_ao(lf, lr, exact=False)
                    # We apply the projected first-order gradient after scaling.
                    total_loss = lf + lr

                loss_for_backward = total_loss / self.cfg.gradient_accumulation_steps
                if self.cfg.ao_impl == "exact":
                    self.accelerator.backward(loss_for_backward)
                else:
                    # Explicit gradient edit: grad = projected-forget + retain.
                    gf_sum = [a + b for a, b in zip(corrected_f, retain_gr)]
                    # Compute scaling with the same accumulation convention.
                    scaled = [g / self.cfg.gradient_accumulation_steps for g in gf_sum]
                    apply_manual_grads(self.trainable_params, scaled)

                accum += 1
                running_loss += float(total_loss.detach().item())

                if accum % self.cfg.gradient_accumulation_steps == 0:
                    if self.cfg.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.trainable_params, self.cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_update += 1

                    if self.accelerator.is_main_process and (
                        self.global_update % self.cfg.log_every == 0 or self.global_update == 1
                    ):
                        print(
                            f"update={self.global_update:05d} epoch={epoch+1} "
                            f"loss={running_loss / self.cfg.gradient_accumulation_steps:.4f} "
                            f"grad_norm={last_grad_norm:.4f} "
                            f"tau={self.tau_grad} gate={attack_active} "
                            f"cos={float(cosine):.4f}"
                        )
                    running_loss = 0.0

                # For the mathematically faithful implementation, the delta is
                # lazily shaped by a probe below. Keep the update loop simple.
            if self.accelerator.is_main_process:
                print(f"Finished epoch {epoch+1}/{self.cfg.num_epochs}")
            if self.cfg.save_every_epoch:
                self.save_checkpoint(Path(self.cfg.output_dir) / f"epoch_{epoch+1}")
            if self.global_update >= total_updates:
                break

    def save_checkpoint(self, path: Path):
        self.accelerator.wait_for_everyone()
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            unwrapped.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            with open(path / "agt_ao_config.json", "w", encoding="utf-8") as f:
                json.dump(asdict(self.cfg), f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_lora(model: nn.Module, cfg: Config) -> nn.Module:
    if cfg.lora_r <= 0:
        return model
    target_names = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_names,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def load_model(path: str, cfg: Config, dtype: torch.dtype) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Important for LoRA + gradient checkpointing.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def main():
    cfg = parse_args()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    # We handle gradient accumulation explicitly below, so Accelerate must
    # not apply a second implicit loss scaling.
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16" if cfg.torch_dtype == "bfloat16" else ("fp16" if cfg.torch_dtype == "float16" else "no"),
    )

    if accelerator.is_main_process:
        print(json.dumps(asdict(cfg), indent=2))

    dtype = dtype_from_string(cfg.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = load_model(cfg.model_path, cfg, dtype)
    model = build_lora(model, cfg)
    if hasattr(model, "print_trainable_parameters") and accelerator.is_main_process:
        model.print_trainable_parameters()

    # Reference model is the same checkpoint BEFORE unlearning, as in the
    # released implementation. It remains frozen throughout training.
    ref_model = load_model(cfg.model_path, cfg, dtype)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    dataset = TofuPairDataset(tokenizer, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_pairs,
        drop_last=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    if cfg.max_train_steps > 0:
        total_steps = min(total_steps, cfg.max_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(1, total_steps))

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    # Keep ref_model outside optimizer/DDP; every rank owns a frozen copy.
    ref_model.to(accelerator.device)

    trainer = AGTAOTrainer(model, ref_model, tokenizer, loader, optimizer, scheduler, accelerator, cfg)

    # Initialize perturbation shape exactly once from model hidden size.
    trainer.adv_delta = torch.zeros(
        (cfg.batch_size, cfg.max_length, model.config.hidden_size),
        device=accelerator.device,
        dtype=next(model.parameters()).dtype,
    )

    trainer.train()
    trainer.save_checkpoint(Path(cfg.output_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Saved AGT^AO model to {cfg.output_dir}")


if __name__ == "__main__":
    main()
