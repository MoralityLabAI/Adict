#!/usr/bin/env python
"""Convert storyworld trace JSONL to Addict Probe training tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.train_data import save_training_tensors, traces_to_training_tensors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export training tensors from trace logs")
    p.add_argument("--trace", required=True, help="Input trace JSONL")
    p.add_argument("--out", default="out/datasets/storyworld_train.npz", help="Output NPZ path")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x, labels = traces_to_training_tensors(args.trace, window=args.window, stride=args.stride)
    save_training_tensors(args.out, x, labels)
    print(
        json.dumps(
            {
                "out": args.out,
                "x_shape": list(x.shape),
                "labels": {k: int(v.shape[0]) for k, v in labels.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
