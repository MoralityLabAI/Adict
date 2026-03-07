#!/usr/bin/env python
"""Train and export Addict lexica from trace rows for runtime diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.probes.dictionary_probe import train_dictionary_probe
from addict_probe.trace_schema import DEFAULT_LAYOUT, read_trace_jsonl, row_to_feature


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Addict lexica from trace JSONL")
    p.add_argument("--trace", required=True, help="Input trace JSONL")
    p.add_argument("--out-npz", default="out/lexica/addict_lexica.npz")
    p.add_argument("--target", default="moral_score_proxy")
    p.add_argument("--dict-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_trace_jsonl(args.trace)
    if not rows:
        raise SystemExit("No rows in trace; cannot train lexica.")

    x = np.stack([row_to_feature(r, layout=DEFAULT_LAYOUT) for r in rows], axis=0).astype(np.float32)
    y = np.asarray([float(r.get(args.target, 0.0)) for r in rows], dtype=np.float32)

    model = train_dictionary_probe(
        x=x,
        y=y,
        dict_size=args.dict_size,
        top_k=args.top_k,
        seed=args.seed,
    )

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        dictionary=model.encoder.dictionary.astype(np.float32),
        head_w=model.head.w.astype(np.float32),
        head_b=np.asarray([float(model.head.b)], dtype=np.float32),
        top_k=np.asarray([int(args.top_k)], dtype=np.int32),
        feat_dim=np.asarray([int(x.shape[1])], dtype=np.int32),
        target=np.asarray([args.target]),
    )

    pred = model.predict(x)
    mse = float(np.mean((pred.reshape(-1) - y.reshape(-1)) ** 2))

    print(
        json.dumps(
            {
                "rows": int(len(rows)),
                "target": args.target,
                "lexica_npz": str(out),
                "dict_size": int(args.dict_size),
                "top_k": int(args.top_k),
                "train_mse": mse,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
