#!/usr/bin/env python
"""Run Adict + ANE-sidecar + TinyLora routing calibration loop scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.hybrid.feature_sets import resolve_feature_set
from addict_probe.hybrid.loop import HybridLoopConfig, run_hybrid_calibration_loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hybrid Adict+ANE+TinyLora loop scaffold")
    p.add_argument("--trace", required=True, help="Trace JSONL from storyworld harness")
    p.add_argument("--out-json", required=True)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--tap-layers", default="16,24")
    p.add_argument("--adapter-names", default="general_storyworld,moral_conflict,powret_helper")
    p.add_argument("--feature-config", default="", help="Optional JSON with named target/control feature sets")
    p.add_argument("--feature-set", default="", help="Feature-set name inside --feature-config")
    p.add_argument("--target-features", default="5,12")
    p.add_argument("--control-features", default="18,25")
    p.add_argument("--late-layer-floor", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapters = [x.strip() for x in args.adapter_names.split(",") if x.strip()]
    if args.feature_config:
        feature_set = resolve_feature_set(args.feature_config, args.feature_set or None)
        tap_layers = list(feature_set.tap_layers)
        target_features = list(feature_set.target_feature_ids)
        control_features = list(feature_set.control_feature_ids)
        late_layer_floor = int(feature_set.late_layer_floor)
    else:
        tap_layers = [int(x.strip()) for x in args.tap_layers.split(",") if x.strip()]
        target_features = [int(x.strip()) for x in args.target_features.split(",") if x.strip()]
        control_features = [int(x.strip()) for x in args.control_features.split(",") if x.strip()]
        late_layer_floor = int(args.late_layer_floor)
    cfg = HybridLoopConfig(
        trace_path=args.trace,
        steps=args.steps,
        batch_size=args.batch_size,
        tap_layers=tap_layers,
        adapter_names=adapters,
        target_feature_ids=target_features,
        control_feature_ids=control_features,
        late_layer_floor=late_layer_floor,
        seed=args.seed,
    )
    report = run_hybrid_calibration_loop(cfg)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
