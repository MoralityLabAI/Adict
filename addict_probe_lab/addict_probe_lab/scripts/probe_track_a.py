#!/usr/bin/env python
"""Linear-probe recovery of planted variables in the Track A model, with controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.splits import assert_group_disjoint, grouped_split
from addict_probe.tracka.probing import (
    RidgeProbe,
    covariance_directions,
    fit_selected,
    permute_within_step,
    r2_score,
)

SLOT_TARGETS = ("v_slot", "q_slot", "h_slot", "z_slot")
# Residual positions probed: the one that predicts the choice and the one that predicts the tag.
POSITION_KEYS = ("ch_pos", "tag_pos")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe the Track A model for planted variables")
    p.add_argument("--run-dir", default="out/track_a/run1")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from addict_probe.tracka.evaluate import imitation_metrics, model_outputs, role_variables, step_mask
    from addict_probe.tracka.model import TinyGPT

    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT.load(run_dir / "model.pt", device)
    max_len = json.loads((run_dir / "train_report.json").read_text(encoding="utf-8"))["max_len"]
    data = dict(np.load(run_dir / "eval.npz"))
    data["tokens"] = data["tokens"][:, :max_len]
    sites = list(range(model.n_sites))
    torch.manual_seed(args.seed + 1)
    untrained = TinyGPT(model.cfg).to(device)

    m = step_mask(data)
    episode = np.broadcast_to(np.arange(m.shape[0])[:, None], m.shape)[m]
    step = np.broadcast_to(np.arange(m.shape[1])[None, :], m.shape)[m]
    split = grouped_split([int(e) for e in episode], seed=args.seed)
    assert_group_disjoint([int(e) for e in episode], split)
    # Probe targets for every existing step; all describe the state before the step's choice.
    targets = {name: data[name][m] for name in SLOT_TARGETS}
    targets.update({name: values[m][:, None] for name, values in role_variables(data).items()})
    rng = np.random.default_rng(args.seed)

    report = {"rows": {k: int(len(v)) for k, v in split.items()}, "positions": {}}
    directions: Dict[str, np.ndarray] = {}
    for key in POSITION_KEYS:
        outputs = model_outputs(model, data, resid_sites=sites, resid_key=key)
        outputs_untrained = model_outputs(untrained, data, resid_sites=sites, resid_key=key)
        report.setdefault("imitation", imitation_metrics(data, outputs))
        key_report = {}
        for site in sites:
            acts = outputs[f"resid_{site}"][m]
            acts_untrained = outputs_untrained[f"resid_{site}"][m]
            x = {name: acts[idx] for name, idx in split.items()}
            x_untrained = {name: acts_untrained[idx] for name, idx in split.items()}
            site_report = {}
            for name, y_all in targets.items():
                y = {s: y_all[idx] for s, idx in split.items()}
                probe, scores = fit_selected(x, y)
                # Control: targets reassigned across training episodes, step index kept.
                y_perm = permute_within_step(y["train"], episode[split["train"]], step[split["train"]], rng)
                control = RidgeProbe.fit(x["train"], y_perm, probe.lam)
                _, untrained_scores = fit_selected(x_untrained, y)
                site_report[name] = {
                    "r2": scores,
                    "lam": probe.lam,
                    "position_control_r2": r2_score(y["test"], control.predict(x["test"])),
                    "untrained_model_r2": untrained_scores["test"],
                }
                directions[f"ridge_{key}_site{site}_{name}"] = probe.directions()
                directions[f"cov_{key}_site{site}_{name}"] = covariance_directions(x["train"], y["train"])
            key_report[str(site)] = site_report
        report["positions"][key] = key_report

    np.savez(run_dir / "probe_directions.npz", **directions)
    np.savez(run_dir / "probe_split.npz", **split)
    (run_dir / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("imitation:", json.dumps({k: round(v, 4) for k, v in report["imitation"].items()}))
    for key in POSITION_KEYS:
        print(f"held-out R^2 at {key} by residual site: trained (position control / untrained model)")
        for name in targets:
            cells = []
            for site in sites:
                r = report["positions"][key][str(site)][name]
                cells.append(
                    f"{r['r2']['test']:5.2f} ({r['position_control_r2']:5.2f}/{r['untrained_model_r2']:5.2f})"
                )
            print(f"{name:<9} " + "  ".join(cells))


if __name__ == "__main__":
    main()
