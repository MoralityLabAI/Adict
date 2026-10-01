#!/usr/bin/env python
"""Build the shared acquisition-prefix pool with grouped splits and a sealed-test manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.persistence.env import FAMILIES, EnvSpec
from addict_probe.persistence.indices import T_ACQ_CHOICES
from addict_probe.persistence.prefixes import build_prefix
from addict_probe.persistence.splits import assert_group_disjoint, grouped_split, write_manifest

OOD_T_ACQ = (32, 48)
N_OPTIONS = (2, 3, 4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build shared acquisition prefixes")
    p.add_argument("--out-dir", default="out/persistence/d_forced")
    p.add_argument("--n-per-family", type=int, default=1000)
    p.add_argument("--n-ood", type=int, default=500, help="Records per held-out split")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def make_records(n: int, seed_base: int, vocab_split: str, t_acq_choices, rng) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        spec = EnvSpec(
            family=FAMILIES[i % len(FAMILIES)],
            n_options=int(rng.choice(N_OPTIONS)),
            t_acq=int(rng.choice(t_acq_choices)),
            vocab_split=vocab_split,
            seed=seed_base + i,
        )
        rows.append(build_prefix(spec))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # Disjoint environment-seed ranges per split family.
    base = args.seed * 1_000_000

    pool = make_records(args.n_per_family * len(FAMILIES), base, "train", T_ACQ_CHOICES, rng)
    groups = [row["env_seed"] for row in pool]
    split = grouped_split(groups, names=("train", "val", "test_id"), seed=args.seed)
    assert_group_disjoint(groups, split)

    files: Dict[str, Path] = {}
    for name, idx in split.items():
        files[name] = out_dir / f"{name}.jsonl"
        write_jsonl(files[name], [pool[int(i)] for i in idx])
    files["ood_vocab"] = out_dir / "ood_vocab.jsonl"
    write_jsonl(files["ood_vocab"], make_records(args.n_ood, base + 500_000, "test", T_ACQ_CHOICES, rng))
    files["ood_length"] = out_dir / "ood_length.jsonl"
    write_jsonl(files["ood_length"], make_records(args.n_ood, base + 700_000, "train", OOD_T_ACQ, rng))

    manifest = write_manifest(
        out_dir / "manifest.json",
        files,
        sealed=("test_id", "ood_vocab", "ood_length"),
        meta={"seed": args.seed, "families": list(FAMILIES), "split_unit": "env_seed"},
    )
    print(json.dumps({k: v["records"] for k, v in manifest["files"].items()}, indent=2))


if __name__ == "__main__":
    main()
