#!/usr/bin/env python
"""Post-train pipeline: lexica training + adapter/base benchmark harness."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.config import AppConfig
from addict_probe.harness.runner import run_storyworld_batch
from addict_probe.evals.storyworld_eval import evaluate_storyworld_trace
from addict_probe.evals.ripe_like_eval import ripe_like_internal_eval
from addict_probe.trace_schema import read_trace_jsonl


def wait_for_path(path: str, timeout_min: int) -> bool:
    if not path:
        return True
    p = Path(path)
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        if p.exists():
            return True
        time.sleep(20)
    return p.exists()


def run_batch(
    cfg: AppConfig, model_path: str, trace_path: str, lexica_npz: str = "", adapter_path: str = ""
) -> dict:
    cfg2 = AppConfig.from_dict(cfg.to_dict())
    cfg2.backend.model_name_or_path = model_path
    cfg2.backend.adapter_path = adapter_path
    cfg2.trace.trace_path = trace_path
    cfg2.trace.lexica_npz_path = lexica_npz
    summary = run_storyworld_batch(cfg2)
    return {
        "trace_path": summary.trace_path,
        "episodes": summary.episodes,
        "rows": summary.rows,
        "scenarios": summary.scenarios,
    }


def train_lexica_from_trace(trace_path: str, out_npz: str) -> dict:
    import numpy as np
    from addict_probe.probes.dictionary_probe import train_dictionary_probe
    from addict_probe.trace_schema import DEFAULT_LAYOUT, row_to_feature

    rows = read_trace_jsonl(trace_path)
    x = np.stack([row_to_feature(r, layout=DEFAULT_LAYOUT) for r in rows], axis=0).astype(np.float32)
    y = np.asarray([float(r.get("moral_score_proxy", 0.0)) for r in rows], dtype=np.float32)
    model = train_dictionary_probe(x, y, dict_size=64, top_k=4, seed=7)

    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        dictionary=model.encoder.dictionary.astype(np.float32),
        head_w=model.head.w.astype(np.float32),
        head_b=np.asarray([float(model.head.b)], dtype=np.float32),
        top_k=np.asarray([4], dtype=np.int32),
        feat_dim=np.asarray([int(x.shape[1])], dtype=np.int32),
        target=np.asarray(["moral_score_proxy"]),
    )
    return {"lexica_npz": str(out), "rows": int(len(rows))}


def eval_trace(path: str) -> dict:
    return {
        **evaluate_storyworld_trace(path),
        **ripe_like_internal_eval(path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run post-train lexica + benchmark pipeline")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--adapter-model", default="", help="Path to trained merged adapter model (optional)")
    p.add_argument("--adapter-path", default="", help="Path to LoRA adapter for base model (optional)")
    p.add_argument("--base-model", required=True, help="Path to base 2B model")
    p.add_argument("--wait-for", default="", help="Optional path to wait for before starting")
    p.add_argument("--wait-timeout-min", type=int, default=180)
    p.add_argument("--out-dir", default="out/posttrain")
    p.add_argument("--primer-episodes", type=int, default=1)
    p.add_argument("--bench-episodes", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter_model and not args.adapter_path:
        raise SystemExit("Provide one of --adapter-model or --adapter-path")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wait_for_path(args.wait_for, args.wait_timeout_min):
        raise SystemExit(f"Timed out waiting for: {args.wait_for}")

    cfg = AppConfig.from_json(args.config)
    cfg.harness.max_turns = args.max_turns

    cfg.harness.episodes_per_scenario = args.primer_episodes
    primer_trace = str(out_dir / "primer_base_trace.jsonl")
    primer_summary = run_batch(cfg, args.base_model, primer_trace, lexica_npz="", adapter_path="")

    lexica_npz = str(out_dir / "addict_lexica.npz")
    lexica_summary = train_lexica_from_trace(primer_trace, lexica_npz)

    cfg.harness.episodes_per_scenario = args.bench_episodes
    base_trace = str(out_dir / "bench_base_trace.jsonl")
    adapter_trace = str(out_dir / "bench_adapter_trace.jsonl")

    base_summary = run_batch(cfg, args.base_model, base_trace, lexica_npz=lexica_npz, adapter_path="")
    adapter_model = args.adapter_model if args.adapter_model else args.base_model
    adapter_summary = run_batch(
        cfg, adapter_model, adapter_trace, lexica_npz=lexica_npz, adapter_path=args.adapter_path
    )

    base_eval = eval_trace(base_trace)
    adapter_eval = eval_trace(adapter_trace)

    keys = [
        "moral_mean",
        "consistency_mean",
        "usefulness_mean",
        "rubric_total",
        "coherence_mean",
        "action_principle_alignment_mean",
        "ripe_like_score",
    ]
    delta = {}
    for k in keys:
        if k in base_eval and k in adapter_eval:
            delta[k] = float(adapter_eval[k]) - float(base_eval[k])

    report = {
        "primer": primer_summary,
        "lexica": lexica_summary,
        "base_run": base_summary,
        "adapter_run": adapter_summary,
        "base_eval": {k: base_eval.get(k) for k in keys},
        "adapter_eval": {k: adapter_eval.get(k) for k in keys},
        "delta_adapter_minus_base": delta,
        "artifacts": {
            "lexica_npz": lexica_npz,
            "primer_trace": primer_trace,
            "base_trace": base_trace,
            "adapter_trace": adapter_trace,
            "adapter_path": args.adapter_path,
        },
    }

    out_json = out_dir / "posttrain_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
