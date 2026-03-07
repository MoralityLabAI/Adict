#!/usr/bin/env python
"""Run internal moral-reasoning evaluations over trace JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.evals.ripe_like_eval import ripe_like_internal_eval
from addict_probe.evals.storyworld_eval import evaluate_storyworld_trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate storyworld trace")
    p.add_argument("--trace", required=True, help="Trace JSONL path")
    p.add_argument("--out", default=None, help="Optional output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "storyworld_eval": evaluate_storyworld_trace(args.trace),
        "ripe_like_eval": ripe_like_internal_eval(args.trace),
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
