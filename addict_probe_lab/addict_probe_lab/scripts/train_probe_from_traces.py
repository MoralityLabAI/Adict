#!/usr/bin/env python
"""Train baseline probes from storyworld traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.probes.dictionary_probe import train_dictionary_probe
from addict_probe.probes.linear_probe import LinearProbe
from addict_probe.probes.temporal_probe import TemporalProbe
from addict_probe.trace_schema import split_indices
from addict_probe.train import DenseBaseline
from addict_probe.train_data import traces_to_training_tensors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train probe baselines from traces")
    p.add_argument("--trace", required=True, help="Input trace JSONL")
    p.add_argument("--method", choices=["dense", "linear", "temporal", "dictionary"], default="temporal")
    p.add_argument(
        "--target",
        choices=["next_moral_score", "future_consistency_delta", "intervention_needed", "future_usefulness_delta"],
        default="next_moral_score",
    )
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def mse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((yhat.reshape(-1) - y.reshape(-1)) ** 2)) if y.size else float("nan")


def main() -> None:
    args = parse_args()
    x, labels = traces_to_training_tensors(args.trace, window=args.window, stride=args.stride)
    y = labels[args.target]

    n = x.shape[0]
    idx = split_indices(n=n, seed=args.seed)
    tr = idx["train"]
    va = idx["val"]
    te = idx["test"]

    if len(tr) == 0:
        raise SystemExit("No training windows found. Generate more episodes or reduce window size.")

    report = {
        "method": args.method,
        "target": args.target,
        "x_shape": list(x.shape),
    }

    if args.method == "dense":
        xtr = x[tr].reshape(len(tr), -1)
        xva = x[va].reshape(len(va), -1) if len(va) else np.zeros((0, xtr.shape[1]), dtype=np.float32)
        xte = x[te].reshape(len(te), -1) if len(te) else np.zeros((0, xtr.shape[1]), dtype=np.float32)
        model = DenseBaseline.fit(xtr, y[tr])
        report["val_mse"] = mse(model.predict(xva), y[va]) if len(va) else float("nan")
        report["test_mse"] = mse(model.predict(xte), y[te]) if len(te) else float("nan")
    elif args.method == "linear":
        x_last = x[:, -1, :]
        model = LinearProbe.fit(x_last[tr], y[tr])
        report["val_mse"] = mse(model.predict(x_last[va]), y[va]) if len(va) else float("nan")
        report["test_mse"] = mse(model.predict(x_last[te]), y[te]) if len(te) else float("nan")
    elif args.method == "temporal":
        model = TemporalProbe.fit(x[tr], y[tr])
        report["val_mse"] = mse(model.predict(x[va]), y[va]) if len(va) else float("nan")
        report["test_mse"] = mse(model.predict(x[te]), y[te]) if len(te) else float("nan")
    else:
        x_last = x[:, -1, :]
        model = train_dictionary_probe(x_last[tr], y[tr], dict_size=64, top_k=4, seed=args.seed)
        report["val_mse"] = mse(model.predict(x_last[va]), y[va]) if len(va) else float("nan")
        report["test_mse"] = mse(model.predict(x_last[te]), y[te]) if len(te) else float("nan")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
