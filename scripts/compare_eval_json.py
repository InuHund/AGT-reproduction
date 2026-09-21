#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def main():
    p = argparse.ArgumentParser(description="Compare two AGT^AO evaluation JSON files")
    p.add_argument("before")
    p.add_argument("after")
    a = p.parse_args()
    before = json.loads(open(a.before).read())
    after = json.loads(open(a.after).read())
    metrics = ("mean_nll_per_token", "ppl", "mean_target_logprob", "mean_rougeL")
    for split in ("forget", "retain"):
        print(f"[{split}]")
        for metric in metrics:
            b = before[split][metric]
            c = after[split][metric]
            print(f"  {metric}: before={b:.6g} after={c:.6g} delta={c-b:+.6g}")


if __name__ == "__main__":
    main()
