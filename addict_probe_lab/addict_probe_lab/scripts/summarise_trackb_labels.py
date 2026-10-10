#!/usr/bin/env python
"""Behavioural summary of Track B labels for one variant: is the persistence label usable?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.splits import open_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise Track B behavioural labels")
    p.add_argument("--labels", required=True, help="labels_<variant>_<split>.jsonl")
    p.add_argument("--pool-dir", default="out/persistence/d_forced_qwen")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="")
    return p.parse_args()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def main() -> None:
    args = parse_args()
    with open(args.labels, encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    with open_split(Path(args.pool_dir) / "manifest.json", args.split).open(encoding="utf-8") as f:
        records = {r["prefix_id"]: r for r in (json.loads(line) for line in f if line.strip())}
    rows = [(lab, records[lab["prefix_id"]]) for lab in labels]
    get = lambda key: np.asarray([lab[key] for lab, _ in rows], dtype=np.float64)
    chance = np.asarray([1.0 / rec["n_options"] for _, rec in rows])
    acq_rate = np.asarray([np.mean(rec["target_selected"]) for _, rec in rows])
    last4 = np.asarray([np.mean(rec["target_selected"][-4:]) for _, rec in rows])

    pi = get("PI")
    p_m0, p_m1, p_mv1, p_mv2 = get("first_p_M0"), get("first_p_M1"), get("first_p_MV1"), get("first_p_MV2")
    report = {
        "n_prefixes": len(rows),
        "valid_mass_min": float(get("valid_mass_min").min()),
        "valid_mass_mean": float(get("valid_mass_mean").mean()),
        "PI": {"mean": float(pi.mean()), "sd": float(pi.std()), "quartiles": np.quantile(pi, [0.25, 0.5, 0.75]).tolist()},
        "persist": {c: float(get(f"persist_{c}").mean()) for c in ("M1", "M2", "M3")},
        "first_step_target_p": {
            "M0 (no notice)": float(p_m0.mean()),
            "stated 0 points (M1)": float(p_m1.mean()),
            "stated 1 point": float(p_mv1.mean()),
            "stated 2 points": float(p_mv2.mean()),
            "equalised (MEQ)": float(get("first_p_MEQ").mean()),
            "novel cue (M6)": float(get("first_p_M6").mean()),
            "prohibited, audit on": float(get("first_p_M7on").mean()),
            "prohibited, audit off": float(get("first_p_M7off").mean()),
        },
        "monotone_in_stated_value": float(np.mean((p_m1 <= p_mv1) & (p_mv1 <= p_mv2))),
        "chance_mean": float(chance.mean()),
        "corr_PI_with_acquisition_target_rate": corr(pi, acq_rate),
        "corr_PI_with_last4_target_rate": corr(pi, last4),
        "corr_PI_with_first_p_M0": corr(pi, p_m0),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
