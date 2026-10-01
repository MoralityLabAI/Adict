#!/usr/bin/env python
"""Where does each factor sit at the choice position? Whole-activation swaps per component.

For every residual site, every mid-block site and every attention-head output
at the choice position, the activation is swapped in from the input that
differs from the base in one factor only. The share of the counterfactual
shift that the swap reproduces is what that component carries. No training is
involved, so this runs on the CPU.

With `--scripts altswap` the action factor is the exchange of the two
equivalent non-target options, which changes the choice kernel and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.yoked import FACTORS, SCRIPT_SOURCES, generate_yoked_dataset

YOKED_SEED_START = 7_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Component-level whole-activation swaps on the Track A model")
    p.add_argument("--run-dir", default="out/track_a/run2")
    p.add_argument("--factor", choices=list(FACTORS), default="notice")
    p.add_argument("--scripts", choices=list(SCRIPT_SOURCES), default="altswap")
    p.add_argument("--n", type=int, default=1500, help="Number of yoked sets")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from addict_probe.tracka.das import kl, prepare_many, single_factor_ceiling
    from addict_probe.tracka.model import Site, TinyGPT, site_name

    run_dir = Path(args.run_dir)
    model = TinyGPT.load(run_dir / "model.pt", args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    data = generate_yoked_dataset(YOKED_SEED_START, args.n, factor=args.factor, scripts=args.scripts)

    n_layer, n_head = model.cfg.n_layer, model.cfg.n_head
    sites: List[Site] = []
    for i in range(1, n_layer + 1):
        sites += [("head", i, h) for h in range(n_head)] + [i - 0.5, i]
    prepared = prepare_many(model, data, sites, "choice", args.device)

    any_y = prepared[sites[0]]
    base, other_action, other_value = any_y.logp[:, 0], any_y.logp[:, 2], any_y.logp[:, 1]
    report: Dict[str, Dict] = {
        "args": vars(args),
        "divergence_between_corners": {
            "action": float(kl(other_action, base).mean()),
            "value": float(kl(other_value, base).mean()),
        },
        "sites": {},
    }
    print(
        f"{args.n} yoked sets, factor={args.factor}, scripts={args.scripts}; mean divergence between corners: "
        f"action {report['divergence_between_corners']['action']:.3f}, "
        f"value {report['divergence_between_corners']['value']:.3f}"
    )
    print(f"{'site':<10}{'action':>8}{'IIA':>6}{'value':>8}{'IIA':>6}")
    for site in sites:
        c = single_factor_ceiling(model, prepared[site], site)
        report["sites"][site_name(site)] = c
        print(
            f"{site_name(site):<10}{c['ceiling_action']:>8.2f}{c['ceiling_iia_action']:>6.2f}"
            f"{c['ceiling_payoff']:>8.2f}{c['ceiling_iia_payoff']:>6.2f}",
            flush=True,
        )
    out = run_dir / f"components_{args.factor}_{args.scripts}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
