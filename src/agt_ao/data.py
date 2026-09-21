from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset


@dataclass
class BatchPair:
    forget: Dict[str, torch.Tensor]
    retain: Dict[str, torch.Tensor]
    raw_forget: Dict[str, Any] | None = None
    raw_retain: Dict[str, Any] | None = None


def _question_answer(example: Dict[str, Any]) -> tuple[str, str]:
    q = example.get("question") or example.get("prompt") or example.get("input")
    a = example.get("answer") or example.get("response") or example.get("output")
    if q is None or a is None:
        raise KeyError(f"Could not find question/answer fields in example keys={list(example.keys())}")
    return str(q), str(a)


def tokenize_qa(tokenizer, q: str, a: str, max_length: int) -> Dict[str, torch.Tensor]:
    prompt = q
    if getattr(tokenizer, "chat_template", None):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = q

    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    answer_ids = tokenizer(a, add_special_tokens=False, truncation=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    ids = (prompt_ids + answer_ids + eos)[:max_length]
    prompt_len = min(len(prompt_ids), len(ids))
    input_ids = ids
    labels = [-100] * prompt_len + ids[prompt_len:]
    labels = labels[:len(input_ids)]
    attention = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_examples(batch, tokenizer, max_length: int):
    items = []
    for ex in batch:
        q, a = _question_answer(ex)
        items.append(tokenize_qa(tokenizer, q, a, max_length=max_length))
    max_len = max(x["input_ids"].numel() for x in items)
    pad = tokenizer.pad_token_id
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in items:
        n = max_len - item["input_ids"].numel()
        out["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, n), value=pad))
        out["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, n), value=0))
        out["labels"].append(torch.nn.functional.pad(item["labels"], (0, n), value=-100))
    return {k: torch.stack(v, dim=0) for k, v in out.items()}


def build_tofu_loaders(
    tokenizer,
    dataset_name: str,
    forget_split: str,
    retain_split: str,
    batch_size: int,
    max_length: int,
    rank: int,
    world_size: int,
    seed: int,
):
    forget_ds = load_dataset(dataset_name, forget_split, split="train")
    retain_ds = load_dataset(dataset_name, retain_split, split="train")

    forget_sampler = DistributedSampler(
        forget_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False
    )
    retain_sampler = DistributedSampler(
        retain_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed + 1, drop_last=False
    )

    collate = lambda batch: collate_examples(batch, tokenizer, max_length=max_length)
    forget_loader = DataLoader(
        forget_ds, batch_size=batch_size, sampler=forget_sampler, collate_fn=collate, pin_memory=True
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=batch_size, sampler=retain_sampler, collate_fn=collate, pin_memory=True
    )
    return forget_ds, retain_ds, forget_loader, retain_loader, forget_sampler, retain_sampler


class PairedLoader:
    def __init__(self, forget_loader, retain_loader):
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader

    def __iter__(self):
        fi = iter(self.forget_loader)
        ri = iter(self.retain_loader)
        max_len = max(len(self.forget_loader), len(self.retain_loader))
        for _ in range(max_len):
            try:
                fb = next(fi)
            except StopIteration:
                fi = iter(self.forget_loader)
                fb = next(fi)
            try:
                rb = next(ri)
            except StopIteration:
                ri = iter(self.retain_loader)
                rb = next(ri)
            yield fb, rb
