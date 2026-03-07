#!/usr/bin/env python
"""Compare two storyworld trace runs and report deltas for key metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.evals.ripe_like_eval import ripe_like_internal_eval
from addict_probe.evals.storyworld_eval import evaluate_storyworld_trace


KEYS = [
    "moral_mean",
    "consistency_mean",
    "usefulness_mean",
    "rubric_total",
    "coherence_mean",
    "action_principle_alignment_mean",
    "ripe_like_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline and candidate trace runs")
    p.add_argument("--run-a", required=True, help="Baseline trace JSONL")
    p.add_argument("--run-b", required=True, help="Candidate trace JSONL")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    a = {**evaluate_storyworld_trace(args.run_a), **ripe_like_internal_eval(args.run_a)}
    b = {**evaluate_storyworld_trace(args.run_b), **ripe_like_internal_eval(args.run_b)}

    delta = {}
    for k in KEYS:
        if k in a and k in b:
            delta[k] = float(b[k]) - float(a[k])

    report = {
        "run_a": args.run_a,
        "run_b": args.run_b,
        "metrics_a": {k: a.get(k) for k in KEYS if k in a},
        "metrics_b": {k: b.get(k) for k in KEYS if k in b},
        "delta_b_minus_a": delta,
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
