#!/usr/bin/env python
"""Measure sensitivity indices for the teacher policies and check regime separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence import (
    REGIME_THETAS,
    BehaviourPolicy,
    assigned_regimes,
    measure_indices,
)

COLUMNS = ["b0", "beta", "S_stated", "S_exp", "BP", "OF", "RF", "EXT", "CG", "kappa", "AG", "RAD", "VQ", "PI"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime manipulation check on teacher policies")
    p.add_argument("--family", choices=["F1", "F2"], default="F1")
    p.add_argument("--n-prefixes", type=int, default=200)
    p.add_argument("--n-rollouts", type=int, default=4)
    p.add_argument("--n-options", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="out/persistence/teacher_indices.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    factories = {
        code: (lambda k, th=theta: BehaviourPolicy(th, k)) for code, theta in REGIME_THETAS.items()
    }
    indices = measure_indices(
        factories,
        n_prefixes=args.n_prefixes,
        n_rollouts=args.n_rollouts,
        n_options=args.n_options,
        family=args.family,
        seed=args.seed,
    )

    print("policy     " + " ".join(f"{c:>8}" for c in COLUMNS) + "  assigned")
    report = {"args": vars(args), "policies": {}}
    failures = []
    for name, ix in indices.items():
        assigned = assigned_regimes(ix)
        print(f"{name:<10} " + " ".join(f"{ix[c]:8.2f}" for c in COLUMNS) + "  " + ",".join(assigned))
        report["policies"][name] = {"indices": ix, "assigned": assigned}
        expected = ["RMAX"] if name == "reference" else [name]
        if assigned != expected:
            failures.append(f"{name}: expected {expected}, got {assigned}")
    # Regions are defined on F1. In F2 a novel cue is ambiguous for every
    # value-sensitive policy, so CG is not diagnostic there and nothing is asserted.
    enforce = args.family == "F1"
    report["failures"] = failures if enforce else []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not enforce:
        print(f"indices only (regions are defined on F1); wrote {out}")
    elif failures:
        raise SystemExit("regime check failed:\n  " + "\n  ".join(failures))
    else:
        print(f"all {len(indices)} policies fall in their own region only; wrote {out}")


if __name__ == "__main__":
    main()
