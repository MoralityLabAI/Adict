#!/usr/bin/env python
"""Sensitivity indices and regime assignment for a Track B model variant (gate G1).

Uses the same forced prefixes, manipulations and regime regions as the teacher
check (run_regime_manipulation_check.py), with the language model as the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence import assigned_regimes, measure_indices

COLUMNS = ["b0", "beta", "S_stated", "S_exp", "BP", "OF", "RF", "EXT", "CG", "kappa", "AG", "RAD", "VQ", "PI"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure regime indices for a Track B variant")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default="", help="LoRA adapter directory; empty for the base model")
    p.add_argument("--variant", default="base")
    p.add_argument("--pool-dir", default="out/persistence/d_forced_qwen", help="Source of the label pool")
    p.add_argument("--n-prefixes", type=int, default=40)
    p.add_argument("--n-rollouts", type=int, default=2)
    p.add_argument("--kappa-samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from addict_probe.trackb.adapters import load_adapter
    from addict_probe.trackb.lm import HookedLM
    from addict_probe.trackb.policy import LMPolicy

    meta = json.loads((Path(args.pool_dir) / "manifest.json").read_text(encoding="utf-8"))["meta"]
    pool = tuple(meta["label_pools"]["train"])
    t0 = time.time()
    lm = HookedLM.load(args.model, device=args.device)
    if args.adapter:
        lm = load_adapter(lm, args.adapter)
    indices = measure_indices(
        {args.variant: lambda k: LMPolicy(lm)},
        n_prefixes=args.n_prefixes,
        n_rollouts=args.n_rollouts,
        n_options=3,
        family="F1",
        seed=args.seed,
        kappa_samples=args.kappa_samples,
        label_pool=pool,
    )
    print("policy     " + " ".join(f"{c:>8}" for c in COLUMNS) + "  assigned")
    report = {"args": vars(args), "elapsed_s": time.time() - t0, "policies": {}}
    for name, ix in indices.items():
        assigned = assigned_regimes(ix)
        print(f"{name:<10} " + " ".join(f"{ix[c]:8.2f}" for c in COLUMNS) + "  " + ",".join(assigned))
        report["policies"][name] = {"indices": ix, "assigned": assigned}
    out = Path(args.out or f"out/trackb/indices/{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
