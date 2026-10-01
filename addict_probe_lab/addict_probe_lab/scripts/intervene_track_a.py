#!/usr/bin/env python
"""Subspace interchange on the Track A model: do probed variables drive the outputs they should?

For a base step and a source step from another episode, the component of the
base residual along a probed direction is replaced by the source's. Two
positions are edited. At the choice position the output of interest is the
choice of the same step; at the tag position it is the tag of the same step,
plus the choice of the next step as a downstream check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.tracka.causal import interchange_factory, pca_basis, random_subspace
from addict_probe.tracka.probing import orthonormal_basis

# Variable -> probe target that defines its direction (role coordinates, one dimension each).
VARIABLES = {"H": "h_target", "V": "v_gap", "D": "d"}
POSITION_KEYS = ("ch_pos", "tag_pos")
MIN_STEP = 2
PCA_DIM = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interchange interventions on the Track A model")
    p.add_argument("--run-dir", default="out/track_a/run1")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    import torch

    from addict_probe.tracka.evaluate import model_outputs, role_variables, step_mask
    from addict_probe.tracka.model import TinyGPT

    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT.load(run_dir / "model.pt", device)
    max_len = json.loads((run_dir / "train_report.json").read_text(encoding="utf-8"))["max_len"]
    full = dict(np.load(run_dir / "eval.npz"))
    full["tokens"] = full["tokens"][:, :max_len]
    directions = dict(np.load(run_dir / "probe_directions.npz"))

    # Interventions use only episodes the probes never saw.
    m_full = step_mask(full)
    episode_of_row = np.broadcast_to(np.arange(m_full.shape[0])[:, None], m_full.shape)[m_full]
    test_eps = np.unique(episode_of_row[np.load(run_dir / "probe_split.npz")["test"]])
    data = {k: v[test_eps] for k, v in full.items()}
    n = len(test_eps)
    rows = np.arange(n)
    n_steps = data["n_steps"].astype(np.int64)
    m = step_mask(data)
    truth = role_variables(data)
    sites = list(range(1, model.n_sites))
    rng = np.random.default_rng(args.seed)

    def target_odds(out: Dict[str, np.ndarray], step: np.ndarray) -> np.ndarray:
        """Log-probability of the target slot minus the mean over the other slots."""
        lp = out["slot_logp"][rows, step]
        t = data["target_slot"][rows, step].astype(np.int64)
        lt = lp[rows, t]
        return lt - (lp.sum(axis=1) - lt) / (lp.shape[1] - 1)

    def slot_p(out: Dict[str, np.ndarray], step: np.ndarray) -> np.ndarray:
        return np.exp(out["slot_logp"][rows, step])

    def tag_logit(out: Dict[str, np.ndarray], step: np.ndarray) -> np.ndarray:
        lp = out["tag_logp"][rows, step]
        return lp[:, 0] - lp[:, 1]

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    report: Dict[str, Dict] = {"n_test_episodes": int(n), "repeats": args.repeats, "positions": {}}
    for key in POSITION_KEYS:
        base_out = model_outputs(model, data, resid_sites=sites, resid_key=key)
        key_report: Dict[str, Dict] = {}
        for site in sites:
            resid = base_out[f"resid_{site}"]
            pca = pca_basis(resid[m], PCA_DIM)
            bases = {"full": np.eye(resid.shape[-1])}
            for var, target in VARIABLES.items():
                for kind in ("ridge", "cov"):
                    bases[f"{var}_{kind}"] = orthonormal_basis(directions[f"{kind}_{key}_site{site}_{target}"])
            # Two control families: uniform within the top principal subspace, and
            # differences of two real activations, which share the variance profile
            # of the activations the way covariance directions do.
            flat = resid[m]
            for i in range(args.n_random):
                bases[f"random_{i}"] = random_subspace(pca, 1, rng)
                pair = rng.choice(flat.shape[0], size=2, replace=False)
                bases[f"actdiff_{i}"] = orthonormal_basis((flat[pair[0]] - flat[pair[1]])[:, None].astype(np.float64))

            acc: Dict[str, Dict[str, List[np.ndarray]]] = {name: {} for name in bases}
            for _ in range(args.repeats):
                # Leave room for a next step so the downstream choice exists.
                base_step = rng.integers(MIN_STEP, n_steps - 1)
                src_ep = (rows + rng.integers(1, n, size=n)) % n  # never the base episode itself
                src_step = rng.integers(MIN_STEP, n_steps[src_ep] - 1)
                choice_step = base_step if key == "ch_pos" else base_step + 1
                positions = data[key][rows, base_step]
                p0, o0, t0 = slot_p(base_out, choice_step), target_odds(base_out, choice_step), tag_logit(base_out, base_step)
                delta = {
                    var: truth[target][src_ep, src_step] - truth[target][rows, base_step]
                    for var, target in VARIABLES.items()
                }
                for name, basis in bases.items():
                    factory = interchange_factory(site, basis, positions, resid[src_ep, src_step], device)
                    out = model_outputs(model, data, edit_factory=factory)
                    t1 = tag_logit(out, base_step)
                    a = acc[name]
                    a.setdefault("tv_choice", []).append(0.5 * np.abs(slot_p(out, choice_step) - p0).sum(axis=1))
                    a.setdefault("tv_tag", []).append(np.abs(sigmoid(t1) - sigmoid(t0)))
                    a.setdefault("d_odds", []).append(target_odds(out, choice_step) - o0)
                    a.setdefault("d_tag", []).append(t1 - t0)
                    for var in VARIABLES:
                        a.setdefault(f"d{var}", []).append(delta[var])

            site_report: Dict[str, Dict[str, float]] = {}
            for name, a in acc.items():
                cat = {k: np.concatenate(v, axis=0) for k, v in a.items()}
                site_report[name] = {
                    "tv_choice": float(cat["tv_choice"].mean()),
                    "tv_tag": float(cat["tv_tag"].mean()),
                    # Do the target's log-odds move with the swapped-in value of the variable?
                    "choice_follows_H": corr(cat["d_odds"], cat["dH"]),
                    "choice_follows_V": corr(cat["d_odds"], cat["dV"]),
                    "choice_follows_D": corr(cat["d_odds"], cat["dD"]),
                    "tag_follows_D": corr(cat["d_tag"], cat["dD"]),
                }
            for family in ("random", "actdiff"):
                names = [f"{family}_{i}" for i in range(args.n_random)]
                summary: Dict[str, float] = {}
                for metric in ("tv_choice", "tv_tag"):
                    values = np.asarray([site_report[nm][metric] for nm in names])
                    summary[f"{metric}_mean"] = float(values.mean())
                    summary[f"{metric}_p95"] = float(np.quantile(values, 0.95))
                    summary[f"{metric}_max"] = float(values.max())
                for nm in names:
                    del site_report[nm]
                site_report[family] = summary
            key_report[str(site)] = site_report
        report["positions"][key] = key_report

    (run_dir / "intervention_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, key_report in report["positions"].items():
        choice_of = "same step" if key == "ch_pos" else "next step"
        print(f"interchange at {key} (choice measured at the {choice_of}, tag at the same step)")
        print(
            f"{'site':<5}{'direction':<10}{'tv_choice':>10}{'tv_tag':>9}"
            f"{'ch~dH':>8}{'ch~dV':>8}{'ch~dD':>8}{'tag~dD':>8}"
        )
        for site, site_report in key_report.items():
            for name, r in site_report.items():
                if name in ("random", "actdiff"):
                    print(
                        f"{site:<5}{name:<10}{r['tv_choice_mean']:>10.3f}{r['tv_tag_mean']:>9.3f}"
                        f"   (max choice {r['tv_choice_max']:.3f}, tag {r['tv_tag_max']:.3f})"
                    )
                else:
                    print(
                        f"{site:<5}{name:<10}{r['tv_choice']:>10.3f}{r['tv_tag']:>9.3f}"
                        f"{r['choice_follows_H']:>8.2f}{r['choice_follows_V']:>8.2f}"
                        f"{r['choice_follows_D']:>8.2f}{r['tag_follows_D']:>8.2f}"
                    )


if __name__ == "__main__":
    main()
