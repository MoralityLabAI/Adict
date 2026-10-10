#!/usr/bin/env python
"""Train a LoRA adapter that imitates one regime's teacher policy (induction method T)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.behaviour_model import REGIME_THETAS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Track B regime adapter by teacher imitation")
    p.add_argument("--model", required=True)
    p.add_argument("--regime", required=True, choices=sorted(REGIME_THETAS))
    p.add_argument("--pool-dir", default="out/persistence/d_forced_qwen", help="Source of the label pool")
    p.add_argument("--out-dir", default="", help="Default: out/trackb/adapters/T_<regime>_s<seed>")
    p.add_argument("--n-train", type=int, default=1600)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--no-value-query", action="store_true", help="Train on choices only (adapters before 2026-10-11)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from addict_probe.trackb.adapters import regime_examples, teacher_nll, train_adapter
    from addict_probe.trackb.lm import HookedLM

    meta = json.loads((Path(args.pool_dir) / "manifest.json").read_text(encoding="utf-8"))["meta"]
    pool = tuple(meta["label_pools"]["train"])
    if not pool:
        raise SystemExit("the prefix pool has no tokenizer-specific label pool; build it with --tokenizer")
    out_dir = Path(args.out_dir or f"out/trackb/adapters/T_{args.regime}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Training and validation episodes use disjoint environment seeds per adapter seed.
    base_seed = 10_000_000 + args.seed * 1_000_000
    vq = not args.no_value_query
    train = regime_examples(args.regime, args.n_train, base_seed, pool, value_query=vq)
    val = regime_examples(args.regime, args.n_val, base_seed + 900_000, pool, value_query=vq)
    floor = teacher_nll(val)
    t0 = time.time()
    lm = HookedLM.load(args.model, device=args.device)
    log_path = out_dir / "train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"regime {args.regime}, seed {args.seed}: {len(train)} train, {len(val)} val episodes; teacher NLL on val {floor:.4f}")
    history = train_adapter(
        lm, train, val, out_dir, epochs=args.epochs, lr=args.lr, rank=args.rank, accumulate=args.accumulate,
        seed=args.seed, log=log,
    )
    report = {
        "args": vars(args),
        "method": "T",
        "teacher_val_nll": floor,
        "history": history,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"saved {out_dir}")


if __name__ == "__main__":
    main()
