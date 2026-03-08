#!/usr/bin/env python
"""Run Adict + ANE-sidecar + TinyLora routing calibration loop scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.hybrid.loop import HybridLoopConfig, run_hybrid_calibration_loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hybrid Adict+ANE+TinyLora loop scaffold")
    p.add_argument("--trace", required=True, help="Trace JSONL from storyworld harness")
    p.add_argument("--out-json", required=True)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--tap-layers", default="16,24")
    p.add_argument("--adapter-names", default="general_storyworld,moral_conflict,powret_helper")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tap_layers = [int(x.strip()) for x in args.tap_layers.split(",") if x.strip()]
    adapters = [x.strip() for x in args.adapter_names.split(",") if x.strip()]
    cfg = HybridLoopConfig(
        trace_path=args.trace,
        steps=args.steps,
        batch_size=args.batch_size,
        tap_layers=tap_layers,
        adapter_names=adapters,
        seed=args.seed,
    )
    report = run_hybrid_calibration_loop(cfg)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

