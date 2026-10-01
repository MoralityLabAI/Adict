#!/usr/bin/env python
"""Learned-subspace interchange on the Track A model with yoked histories (spec C4/C5).

Each yoked set holds four forced histories in one environment: two action
scripts crossed with two levels of a value factor (the points on outcome lines,
or a notice that states the target's points). For a base corner and the
opposite corner as source, swapping the learned "action" subspace should
reproduce the model's own output on the corner with the source's actions and
the base's value information, and the "payoff" subspace the reverse. Learned
subspaces are compared with untrained ones, with probe directions, and with
swapping the whole residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.splits import assert_group_disjoint, grouped_split
from addict_probe.persistence.yoked import FACTORS, SCRIPT_SOURCES, generate_yoked_dataset
from addict_probe.tracka.probing import RidgeProbe, covariance_directions, orthonormal_basis

YOKED_SEED_START = 7_000_000
METRICS = ("recovered_action", "recovered_payoff", "cross_action", "cross_payoff", "iia_action", "iia_payoff")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learned-subspace interchange on the Track A model")
    p.add_argument("--run-dir", default="out/track_a/run1")
    p.add_argument("--factor", choices=list(FACTORS), default="payoff", help="Value factor of the yoked sets")
    p.add_argument("--scripts", choices=list(SCRIPT_SOURCES), default="teacher", help="Source of the action scripts")
    p.add_argument("--n", type=int, default=6000, help="Number of yoked sets")
    p.add_argument("--k", type=int, default=8, help="Subspace dimension for the configuration sweep")
    p.add_argument("--k-sweep", default="1,2,4,8,16")
    p.add_argument("--sweep-config", default="", help="Extra configurations to sweep, e.g. choice_site4")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def probe_targets(data: Dict[str, np.ndarray], where: str) -> Tuple[np.ndarray, np.ndarray]:
    """Own-choice and payoff variables matching the swapped positions, [sets, corners, positions]."""
    h, v = data["h_slot"], data["v_slot"]
    if where == "options":  # each option's own value at its own token
        return h, v
    t = data["target_slot"].astype(np.int64)[:, None, None]
    is_target = np.arange(v.shape[2])[None, None, :] == t
    h_target = np.take_along_axis(h, np.broadcast_to(t, h.shape[:2] + (1,)), axis=2)
    v_gap = np.take_along_axis(v, np.broadcast_to(t, v.shape[:2] + (1,)), axis=2)[..., 0] - np.where(
        is_target, -np.inf, v
    ).max(axis=2)
    return h_target, v_gap[..., None]


def main() -> None:
    args = parse_args()
    import torch

    from addict_probe.tracka.das import (
        SubspacePair,
        evaluate_bases,
        fixed_bases,
        prepare,
        single_factor_ceiling,
        subset,
        train_subspaces,
    )
    from addict_probe.tracka.model import TinyGPT

    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT.load(run_dir / "model.pt", device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    d_model = model.cfg.d_model

    data = generate_yoked_dataset(YOKED_SEED_START, args.n, factor=args.factor, scripts=args.scripts)
    groups = [int(s) for s in data["env_seed"]]
    split = grouped_split(groups, seed=args.seed)
    assert_group_disjoint(groups, split)

    def run_config(where: str, site: int, k: int) -> Dict[str, Dict[str, float]]:
        y = prepare(model, data, site, where, device)
        parts = {name: subset(y, idx) for name, idx in split.items()}
        pair = train_subspaces(model, parts["train"], site, k, steps=args.steps, seed=args.seed)
        with torch.no_grad():
            learned = pair.bases()
            untrained = SubspacePair(d_model, k, seed=args.seed + 100).to(device).bases()
        eye = np.eye(d_model)
        # One-dimensional probe directions fitted on the training sets.
        acts = parts["train"].resid.cpu().numpy().reshape(-1, d_model).astype(np.float64)
        h, v = probe_targets({key: val[split["train"]] for key, val in data.items()}, where)
        h, v = h.reshape(-1, 1), v.reshape(-1, 1)
        ridge = {name: orthonormal_basis(RidgeProbe.fit(acts, t, 1.0).directions()) for name, t in (("h", h), ("v", v))}
        cov = {name: orthonormal_basis(covariance_directions(acts, t)) for name, t in (("h", h), ("v", v))}
        result = {
            "learned_val": evaluate_bases(model, parts["val"], site, learned),
            "learned": evaluate_bases(model, parts["test"], site, learned),
            "untrained": evaluate_bases(model, parts["test"], site, untrained),
            "full_residual": evaluate_bases(model, parts["test"], site, fixed_bases(eye, eye, device)),
            "probe_ridge": evaluate_bases(model, parts["test"], site, fixed_bases(ridge["h"], ridge["v"], device)),
            "probe_cov": evaluate_bases(model, parts["test"], site, fixed_bases(cov["h"], cov["v"], device)),
        }
        return result

    def show(label: str, r: Dict[str, float]) -> None:
        print(f"{label:<34}" + "".join(f"{r[m]:>10.2f}" for m in METRICS), flush=True)

    n_sites = model.n_sites
    # Option and notice tokens precede the choice position, so a swap there needs a later block to matter.
    configs: List[Tuple[str, int]] = [("choice", s) for s in range(1, n_sites)] + [
        ("options", s) for s in range(1, n_sites - 1)
    ]
    if args.factor == "notice":
        # At site 0 the whole-residual swap is a token swap, so it must reproduce the counterfactual exactly.
        configs += [("notice", s) for s in range(0, n_sites - 1)]
    # Results are saved after every configuration and reused on a rerun with the same arguments;
    # on a shared GPU a run can be cut short.
    report_path = run_dir / f"das_report_{args.factor}_{args.scripts}.json"
    # Arguments that only select what to print or sweep do not invalidate saved results.
    identity = {k: v for k, v in vars(args).items() if k not in ("k_sweep", "sweep_config")}
    report: Dict[str, Dict] = {"args": identity, "sets": {k: int(len(v)) for k, v in split.items()}}
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        old = {k: v for k, v in previous.get("args", {}).items() if k not in ("k_sweep", "sweep_config")}
        if old == identity:
            report = {**previous, "args": identity}
    report.setdefault("configs", {})
    report.setdefault("k_sweeps", {})
    old_sweep = report.pop("k_sweep", None)  # earlier layout: one sweep, at the best configuration
    if old_sweep and report.get("best_config"):
        report["k_sweeps"].setdefault(report["best_config"], old_sweep)
    report["k_sweeps"].pop("", None)

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'':<34}" + "".join(f"{m.replace('recovered', 'rec').replace('_action', '_A').replace('_payoff', '_P'):>10}" for m in METRICS))
    for where, site in configs:
        key = f"{where}_site{site}"
        if key not in report["configs"]:
            report["configs"][key] = run_config(where, site, args.k)
            save()
        res = report["configs"][key]
        if "ceiling" not in res:
            y = prepare(model, data, site, where, device)
            res["ceiling"] = single_factor_ceiling(model, subset(y, split["test"]), site)
            res["ceiling_val"] = single_factor_ceiling(model, subset(y, split["val"]), site)
            save()
        c = res["ceiling"]
        print(
            f"{where} site {site}: whole-residual swap of one factor recovers "
            f"action {c['ceiling_action']:.2f} (IIA {c['ceiling_iia_action']:.2f}), "
            f"value {c['ceiling_payoff']:.2f} (IIA {c['ceiling_iia_payoff']:.2f})"
        )
        for name in ("learned", "untrained", "full_residual", "probe_ridge", "probe_cov"):
            show(f"{where} site {site} k={args.k if name in ('learned', 'untrained') else '-'} {name}", res[name])

    def val_score(key: str) -> float:
        # A learned subspace is credited with no more than the site carries (see single_factor_ceiling).
        r, c = report["configs"][key]["learned_val"], report["configs"][key]["ceiling_val"]
        return sum(min(r[f"recovered_{v}"], max(c[f"ceiling_{v}"], 0.0)) for v in ("action", "payoff"))

    best = max(report["configs"], key=val_score)
    report["best_config"] = best
    for sweep_config in dict.fromkeys([best] + [c for c in args.sweep_config.split(",") if c]):
        where, site = sweep_config.split("_site")
        sweep = report["k_sweeps"].setdefault(sweep_config, {})
        chosen = "chosen on validation sets" if sweep_config == best else "requested"
        print(f"dimension sweep at {sweep_config} ({chosen})")
        for k in [int(x) for x in args.k_sweep.split(",")]:
            if str(k) not in sweep:
                res = run_config(where, int(site), k)
                sweep[str(k)] = {"learned": res["learned"], "untrained": res["untrained"]}
                save()
            show(f"k={k} learned", sweep[str(k)]["learned"])
            show(f"k={k} untrained", sweep[str(k)]["untrained"])

    first = next(iter(report["configs"].values()))["learned"]
    print(
        "counterfactual changes the most probable option in "
        f"{first['flip_rate_action']:.0%} of cases for action history and {first['flip_rate_payoff']:.0%} for payoff history"
    )
    save()


if __name__ == "__main__":
    main()
