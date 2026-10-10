#!/usr/bin/env python
"""Known-answer checks on stored Track B activations (spec section 8).

A linear probe at t* must read the cue of the last acquisition step almost
perfectly at some site, and the same probe on cue labels shuffled across
prefixes must sit at chance. Cross-validation folds are grouped by prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.splits import grouped_split, open_split
from addict_probe.tracka.probing import RidgeProbe
from addict_probe.trackb.store import ActivationStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Known-answer checks on the Track B store")
    p.add_argument("--store", default="out/trackb/store")
    p.add_argument("--variant", default="base")
    p.add_argument("--pool-dir", default="out/persistence/d_forced_qwen")
    p.add_argument("--split", default="train")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def last_cue(text: str) -> str:
    return [line for line in text.splitlines() if line.startswith("cue: ")][-1][5:]


def cv_accuracy(x: np.ndarray, y: np.ndarray, groups: List[str], folds: int, lam: float, seed: int) -> float:
    classes = sorted(set(y.tolist()))
    onehot = (y[:, None] == np.asarray(classes)[None, :]).astype(np.float64)
    names = tuple(f"f{i}" for i in range(folds))
    split = grouped_split(groups, fractions=[1.0 / folds] * folds, names=names, seed=seed)
    hits = 0
    for name, test in split.items():
        train = np.setdiff1d(np.arange(len(y)), test)
        probe = RidgeProbe.fit(x[train], onehot[train], lam)
        hits += int(np.sum(probe.predict(x[test]).argmax(axis=1) == onehot[test].argmax(axis=1)))
    return hits / len(y)


def main() -> None:
    args = parse_args()
    store = ActivationStore(args.store)
    with open_split(Path(args.pool_dir) / "manifest.json", args.split).open(encoding="utf-8") as f:
        records = {r["prefix_id"]: r for r in (json.loads(line) for line in f if line.strip())}
    sites = sorted({int(k.split("__s")[1].split("__")[0]) for k in store.keys() if k.startswith(f"{args.variant}__")})
    rng = np.random.default_rng(args.seed)
    report: Dict[str, Dict[str, float]] = {}
    print(f"{'site':>5}{'cue acc':>9}{'shuffled':>10}")
    for site in sites:
        rows, acts = store.read(args.variant, site, "tstar")
        x = acts[:, 0].astype(np.float64)
        ids = [r["prefix_id"] for r in rows]
        y = np.asarray([last_cue(records[i]["text"]) for i in ids])
        chance = float(np.max(np.unique(y, return_counts=True)[1]) / len(y))
        acc = cv_accuracy(x, y, ids, args.folds, args.lam, args.seed)
        shuffled = cv_accuracy(x, rng.permutation(y), ids, args.folds, args.lam, args.seed)
        report[str(site)] = {"cue_accuracy": acc, "shuffled_accuracy": shuffled, "majority_rate": chance}
        print(f"{site:>5}{acc:>9.2f}{shuffled:>10.2f}", flush=True)
    best = max(v["cue_accuracy"] for v in report.values())
    worst_shuffled = max(v["shuffled_accuracy"] for v in report.values())
    majority = next(iter(report.values()))["majority_rate"]
    print(f"best cue accuracy {best:.2f}; largest shuffled accuracy {worst_shuffled:.2f}; majority-class rate {majority:.2f}")
    out = Path(args.store) / f"known_answer_{args.variant}_{args.split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
