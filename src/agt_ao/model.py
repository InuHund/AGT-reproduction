from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


def load_causal_lm(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    return LoadedModel(model=model, tokenizer=tokenizer)


def transformer_blocks(model: Any):
    # Unwrap common distributed/model wrappers before locating the Transformer stack.
    while hasattr(model, "module") and not hasattr(model, "model"):
        model = model.module
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.base_model.model.layers,
        lambda m: m.transformer.h,
    ]
    for getter in candidates:
        try:
            blocks = getter(model)
            if blocks is not None:
                return blocks
        except AttributeError:
            continue
    raise ValueError("Could not find Transformer blocks for the supplied model.")


class LatentAddHook:
    def __init__(self, model: Any, one_based_layer: int, delta: torch.Tensor | None = None):
        self.model = model
        self.layer = one_based_layer
        self.delta = delta
        self.handle = None

    def __enter__(self):
        blocks = transformer_blocks(self.model)
        if not (1 <= self.layer <= len(blocks)):
            raise ValueError(f"perturb_layer={self.layer} outside 1..{len(blocks)}")
        module = blocks[self.layer - 1]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.delta is None:
                self.delta = torch.zeros_like(hidden)
            if self.delta.device != hidden.device:
                self.delta = self.delta.to(hidden.device)
            perturbed = hidden + self.delta
            if isinstance(output, tuple):
                return (perturbed,) + output[1:]
            return perturbed

        self.handle = module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
