from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import _question_answer, tokenize_qa
from .losses import sequence_logprob


def args():
    p = argparse.ArgumentParser(description="Standalone AGT^AO / TOFU evaluation")
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_name", default="locuslab/TOFU")
    p.add_argument("--forget_split", default="forget10")
    p.add_argument("--retain_split", default="retain90")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_eval_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--reference_model_path", default=None, help="Optional pre-unlearning model for before/after deltas")
    p.add_argument("--output_json", default=None)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default="eval")
    return p.parse_args()


def load(path):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device = next(model.parameters()).device
    return model, tok, device


def eval_split(model, tok, device, ds, max_len, max_samples, max_new):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total_nll = 0.0
    total_tokens = 0
    total_logp = 0.0
    rouge_values = []
    n = min(len(ds), max_samples)

    for i in range(n):
        ex = ds[i]
        q, a = _question_answer(ex)
        item = tokenize_qa(tok, q, a, max_len)
        batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
        with torch.no_grad():
            out = model(**batch)
            logp = sequence_logprob(out.logits, batch["labels"])[0].item()
            num = int(batch["labels"][:, 1:].ne(-100).sum().item())
            total_logp += logp
            total_nll += -logp
            total_tokens += num

            prompt = q
            if getattr(tok, "chat_template", None):
                try:
                    prompt = tok.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
            inp = tok(prompt, return_tensors="pt").to(device)
            gen = model.generate(
                **inp,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(
                gen[0][inp["input_ids"].size(1):],
                skip_special_tokens=True,
            )
            rouge_values.append(scorer.score(a, text)["rougeL"].fmeasure)

    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "n": n,
        "mean_nll_per_token": mean_nll,
        "ppl": math.exp(min(mean_nll, 50.0)),
        "mean_target_logprob": total_logp / max(n, 1),
        "mean_rougeL": sum(rouge_values) / max(len(rouge_values), 1),
    }


def flatten(prefix, result):
    return {
        f"{prefix}/nll_per_token": result["mean_nll_per_token"],
        f"{prefix}/ppl": result["ppl"],
        f"{prefix}/target_logprob": result["mean_target_logprob"],
        f"{prefix}/rougeL": result["mean_rougeL"],
    }


def main():
    a = args()
    model, tok, device = load(a.model_path)
    forget = load_dataset(a.dataset_name, a.forget_split, split="train")
    retain = load_dataset(a.dataset_name, a.retain_split, split="train")

    result = {
        "model_path": a.model_path,
        "forget": eval_split(model, tok, device, forget, a.max_length, a.max_eval_samples, a.max_new_tokens),
        "retain": eval_split(model, tok, device, retain, a.max_length, a.max_eval_samples, a.max_new_tokens),
    }

    if a.reference_model_path:
        ref_model, ref_tok, ref_device = load(a.reference_model_path)
        ref_result = {
            "model_path": a.reference_model_path,
            "forget": eval_split(ref_model, ref_tok, ref_device, forget, a.max_length, a.max_eval_samples, a.max_new_tokens),
            "retain": eval_split(ref_model, ref_tok, ref_device, retain, a.max_length, a.max_eval_samples, a.max_new_tokens),
        }
        result["reference"] = ref_result
        result["delta"] = {
            split: {
                metric: result[split][metric] - ref_result[split][metric]
                for metric in ("mean_nll_per_token", "ppl", "mean_target_logprob", "mean_rougeL")
            }
            for split in ("forget", "retain")
        }

    print(json.dumps(result, indent=2))
    if a.output_json:
        Path(a.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output_json).write_text(json.dumps(result, indent=2))

    if a.wandb_project:
        import wandb

        run = wandb.init(project=a.wandb_project, name=a.wandb_run_name, config=vars(a))
        flat = {}
        flat.update(flatten("forget", result["forget"]))
        flat.update(flatten("retain", result["retain"]))
        if "reference" in result:
            flat.update(flatten("reference/forget", result["reference"]["forget"]))
            flat.update(flatten("reference/retain", result["reference"]["retain"]))
            for split in ("forget", "retain"):
                for metric, value in result["delta"][split].items():
                    flat[f"delta/{split}/{metric}"] = value
        run.log(flat)
        run.finish()


if __name__ == "__main__":
    main()
