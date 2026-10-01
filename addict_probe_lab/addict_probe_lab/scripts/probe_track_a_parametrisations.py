#!/usr/bin/env python
"""Where and in which coordinates are the planted per-option variables readable?

Compares linear and one-hidden-layer probes for the same variables under four
parametrisations: displayed slot at the choice position, target role at the
choice position, option label at the choice position, and each option's own
label token in the options line. An untrained copy of the model is the baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.splits import grouped_split
from addict_probe.tracka.probing import fit_label_indexed, fit_selected

PER_OPTION = ("v_slot", "h_slot", "z_slot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare probe parametrisations on the Track A model")
    p.add_argument("--run-dir", default="out/track_a/run1")
    p.add_argument("--sites", default="2,3,4")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from addict_probe.tracka.evaluate import (
        model_outputs,
        option_token_positions,
        role_variables,
        slot_token_ids,
        step_mask,
    )
    from addict_probe.tracka.mlp_probe import mlp_probe_r2
    from addict_probe.tracka.model import TinyGPT

    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained = TinyGPT.load(run_dir / "model.pt", device)
    max_len = json.loads((run_dir / "train_report.json").read_text(encoding="utf-8"))["max_len"]
    data = dict(np.load(run_dir / "eval.npz"))
    data["tokens"] = data["tokens"][:, :max_len]
    sites = [int(s) for s in args.sites.split(",")]
    torch.manual_seed(args.seed + 1)
    models = {"trained": trained, "untrained": TinyGPT(trained.cfg).to(device)}

    m = step_mask(data)
    k = data["order"].shape[2]
    episode = np.broadcast_to(np.arange(m.shape[0])[:, None], m.shape)[m]
    split = grouped_split([int(e) for e in episode], seed=args.seed)
    slot_tok = slot_token_ids(data)[m]
    opt_pos = option_token_positions(data)
    for j in range(k):
        data[f"opt_pos_{j}"] = opt_pos[:, :, j]
    role = role_variables(data)

    def parts(a: np.ndarray) -> Dict[str, np.ndarray]:
        return {s: a[idx] for s, idx in split.items()}

    def mlp(x: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> float:
        try:
            return mlp_probe_r2(x, y, seed=args.seed)
        except RuntimeError:
            # A shared GPU can run out of memory mid-run; the probe is small enough for the CPU.
            torch.cuda.empty_cache()
            return mlp_probe_r2(x, y, seed=args.seed, device="cpu")

    def both(x: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> Dict[str, float]:
        return {"ridge": fit_selected(x, y)[1]["test"], "mlp": mlp(x, y)}

    # Results are written after every (site, model) cell and reused on a rerun.
    report_path = run_dir / "parametrisation_report.json"
    report: Dict[str, Dict] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for model_name, model in models.items():
        todo = [s for s in sites if model_name not in report.get(str(s), {})]
        if not todo:
            continue
        at_choice = model_outputs(model, data, resid_sites=todo, resid_key="ch_pos")
        at_option = [model_outputs(model, data, resid_sites=todo, resid_key=f"opt_pos_{j}") for j in range(k)]
        for site in todo:
            x_choice = parts(at_choice[f"resid_{site}"][m])
            # One row per (step, displayed option), pooled over slots.
            x_option = {
                s: np.concatenate([at_option[j][f"resid_{site}"][m][idx] for j in range(k)]) for s, idx in split.items()
            }
            cell: Dict[str, Dict[str, float]] = {}
            for name in ("v_gap", "h_target"):
                cell[f"role/{name}"] = both(x_choice, parts(role[name][m][:, None]))
            for name in PER_OPTION:
                values = data[name][m]
                cell[f"slot/{name}"] = both(x_choice, parts(values))
                cell[f"label/{name}"] = {
                    "ridge": fit_label_indexed(x_choice, parts(slot_tok), parts(values))[1]["test"]
                }
                y_option = {s: np.concatenate([values[idx, j] for j in range(k)])[:, None] for s, idx in split.items()}
                cell[f"option_token/{name}"] = both(x_option, y_option)
            report.setdefault(str(site), {})[model_name] = cell
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"site {site} {model_name} done", flush=True)

    print("held-out R^2: trained ridge / trained mlp (untrained ridge / untrained mlp)")
    for key in report[str(sites[0])]["trained"]:
        cells = []
        for site in sites:
            t, u = report[str(site)]["trained"][key], report[str(site)]["untrained"][key]
            fmt = lambda r: f"{r['ridge']:5.2f}/" + (f"{r['mlp']:5.2f}" if "mlp" in r else "  -  ")
            cells.append(f"{fmt(t)} ({fmt(u)})")
        print(f"{key:<22}" + "   ".join(cells))


if __name__ == "__main__":
    main()
