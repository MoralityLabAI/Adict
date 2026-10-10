#!/usr/bin/env python
"""Known-answer checks on stored Track B activations (spec section 8).

At t* (the last token of the last outcome line, which is the outcome's tag)
a linear probe must read that tag perfectly at the embedding site, and the
same probe on shuffled tags must sit at chance. The cue of the last step,
three lines earlier, tests whether information is carried forward; with few
prefixes per cue it is underpowered and is reported, not required.
Cross-validation folds are grouped by prefix.
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


def last_tag(text: str) -> str:
    return [line for line in text.splitlines() if line.startswith("outcome: ")][-1].rsplit(" ", 1)[1]


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
    print(f"{'site':>5}{'tag acc':>9}{'shuffled':>10}{'cue acc':>9}{'shuffled':>10}")
    for site in sites:
        rows, acts = store.read(args.variant, site, "tstar")
        x = acts[:, 0].astype(np.float64)
        ids = [r["prefix_id"] for r in rows]
        site_report: Dict[str, float] = {}
        for name, fn in (("tag", last_tag), ("cue", last_cue)):
            y = np.asarray([fn(records[i]["text"]) for i in ids])
            site_report[f"{name}_majority_rate"] = float(np.max(np.unique(y, return_counts=True)[1]) / len(y))
            site_report[f"{name}_accuracy"] = cv_accuracy(x, y, ids, args.folds, args.lam, args.seed)
            site_report[f"{name}_shuffled_accuracy"] = cv_accuracy(
                x, rng.permutation(y), ids, args.folds, args.lam, args.seed
            )
        report[str(site)] = site_report
        r = site_report
        print(
            f"{site:>5}{r['tag_accuracy']:>9.2f}{r['tag_shuffled_accuracy']:>10.2f}"
            f"{r['cue_accuracy']:>9.2f}{r['cue_shuffled_accuracy']:>10.2f}",
            flush=True,
        )
    first = report[str(sites[0])]
    passed = first["tag_accuracy"] >= 0.95 and max(v["tag_shuffled_accuracy"] for v in report.values()) <= first["tag_majority_rate"] + 0.25
    print(
        f"tag at the embedding site {first['tag_accuracy']:.2f} (majority {first['tag_majority_rate']:.2f}); "
        f"best cue {max(v['cue_accuracy'] for v in report.values()):.2f} (majority {first['cue_majority_rate']:.2f}); "
        f"known-answer check {'passed' if passed else 'FAILED'}"
    )
    out = Path(args.store) / f"known_answer_{args.variant}_{args.split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
