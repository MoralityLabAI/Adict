#!/usr/bin/env python
"""Estimate lexica atoms that improve or degrade future moral trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.mechinterp.transition_lens import write_transition_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank lexica atoms by future moral-trajectory lift")
    parser.add_argument("--trace", required=True, help="Trace JSONL to analyze")
    parser.add_argument("--lexica-npz", required=True, help="Lexica NPZ produced from trace rows")
    parser.add_argument("--out-json", default="out/analysis/transition_lens.json")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--activation-floor", type=float, default=0.05)
    parser.add_argument("--top-atoms", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_transition_report(
        trace_path=args.trace,
        lexica_npz=args.lexica_npz,
        out_json=args.out_json,
        horizon=args.horizon,
        activation_floor=args.activation_floor,
        top_atoms=args.top_atoms,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
