#!/usr/bin/env python
"""Collect residual activations and behavioural labels for one model variant on a prefix split.

Resumable: prefixes already stored for the variant are skipped. Each run starts
with a reproducibility check (a second forward pass must reproduce stored
activations within float16 tolerance) and stops if it fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.splits import open_split

POSITIONS = ("tstar", "window")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Track B activations and labels")
    p.add_argument("--model", required=True, help="Checkpoint directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--variant", default="base", help="Name of the model variant (base, sham, an adapter)")
    p.add_argument("--pool-dir", default="out/persistence/d_forced_qwen")
    p.add_argument("--split", default="train")
    p.add_argument("--analysis-id", default="", help="Required for sealed splits")
    p.add_argument("--store", default="out/trackb/store")
    p.add_argument("--sites", default="all", help="'all' or comma-separated residual sites")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--labels", action="store_true", help="Also measure behavioural labels (slow)")
    p.add_argument("--n-rollouts", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from addict_probe.trackb.collect import collect_batch
    from addict_probe.trackb.labels import prefix_labels
    from addict_probe.trackb.lm import HookedLM
    from addict_probe.trackb.store import ActivationStore

    pool = Path(args.pool_dir)
    split_path = open_split(pool / "manifest.json", args.split, analysis_id=args.analysis_id or None)
    with split_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]

    t0 = time.time()
    lm = HookedLM.load(args.model, device=args.device)
    sites: List[int] = list(range(lm.n_sites)) if args.sites == "all" else [int(s) for s in args.sites.split(",")]
    print(f"loaded {args.model} in {time.time() - t0:.0f}s: {lm.n_layers} layers, d_model {lm.d_model}", flush=True)
    store = ActivationStore(args.store)
    done = store.stored_prefix_ids(args.variant, sites[-1], "tstar") if store.keys() else set()
    todo = [r for r in records if r["prefix_id"] not in done]
    labels_path = Path(args.store) / f"labels_{args.variant}_{args.split}.jsonl"
    labelled = set()
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            labelled = {json.loads(line)["prefix_id"] for line in f if line.strip()}
    # Prefixes whose activations were stored before an interrupted run could label them.
    label_only = [r for r in records if args.labels and r["prefix_id"] in done and r["prefix_id"] not in labelled]
    print(f"{len(records)} prefixes in {args.split}, {len(todo)} to collect, {len(label_only)} to label only", flush=True)
    for r in label_only:
        with labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**prefix_labels(lm, r, n_rollouts=args.n_rollouts), "variant": args.variant}) + "\n")

    checked = False
    for lo in range(0, len(todo), args.batch_size):
        batch = todo[lo : lo + args.batch_size]
        acts = collect_batch(lm, batch, sites, POSITIONS)
        if not checked:
            again = collect_batch(lm, batch, sites, POSITIONS)
            for key in acts:
                a, b = acts[key].astype(np.float16).astype(np.float32), again[key].astype(np.float16).astype(np.float32)
                scale = float(np.abs(a).max()) or 1.0
                if float(np.abs(a - b).max()) > 1e-2 * scale:
                    raise SystemExit(f"activations at {key} do not reproduce between forward passes")
            checked = True
        meta = [
            {
                "prefix_id": r["prefix_id"],
                "env_seed": r["env_seed"],
                "variant": args.variant,
                "split": args.split,
                "family": r["family"],
                "n_options": r["n_options"],
                "t_acq": r["t_acq"],
            }
            for r in batch
        ]
        for (site, name), a in acts.items():
            store.write(args.variant, site, name, meta, a)
        if args.labels:
            with labels_path.open("a", encoding="utf-8") as f:
                for r in batch:
                    f.write(json.dumps({**prefix_labels(lm, r, n_rollouts=args.n_rollouts), "variant": args.variant}) + "\n")
        done_n = lo + len(batch)
        rate = (time.time() - t0) / done_n
        print(f"{done_n}/{len(todo)} prefixes, {rate:.1f}s each", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
