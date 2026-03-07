#!/usr/bin/env python
"""Run repeated storyworld episodes and emit JSONL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.config import AppConfig
from addict_probe.harness.runner import run_storyworld_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run storyworld batch harness")
    p.add_argument("--config", default="configs/default.json", help="Path to app config JSON")
    p.add_argument("--trace-path", default=None, help="Override trace JSONL path")
    p.add_argument("--backend", default=None, help="Override backend: qwen_mlx or qwen_transformers")
    p.add_argument("--model-path", default=None, help="Override model path")
    p.add_argument("--adapter-path", default=None, help="Optional adapter path (MLX path)")
    p.add_argument("--episodes-per-scenario", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--enable-think-step", action="store_true", help="Enable bounded intermediate think step")
    p.add_argument("--disable-think-step", action="store_true", help="Disable intermediate think step")
    p.add_argument("--think-max-new-tokens", type=int, default=None)
    p.add_argument("--think-char-cap", type=int, default=None)
    p.add_argument("--lexica-npz", default=None, help="Optional Addict lexica NPZ for runtime diagnostics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig.from_json(args.config)

    if args.trace_path:
        cfg.trace.trace_path = args.trace_path
    if args.backend:
        cfg.backend.backend = args.backend
    if args.model_path:
        cfg.backend.model_name_or_path = args.model_path
    if args.adapter_path is not None:
        cfg.backend.adapter_path = args.adapter_path
    if args.episodes_per_scenario is not None:
        cfg.harness.episodes_per_scenario = args.episodes_per_scenario
    if args.max_turns is not None:
        cfg.harness.max_turns = args.max_turns
    if args.enable_think_step:
        cfg.backend.enable_think_step = True
    if args.disable_think_step:
        cfg.backend.enable_think_step = False
    if args.think_max_new_tokens is not None:
        cfg.backend.think_max_new_tokens = args.think_max_new_tokens
    if args.think_char_cap is not None:
        cfg.backend.think_char_cap = args.think_char_cap
    if args.lexica_npz:
        cfg.trace.lexica_npz_path = args.lexica_npz

    Path(cfg.trace.trace_path).parent.mkdir(parents=True, exist_ok=True)
    summary = run_storyworld_batch(cfg)
    print(
        json.dumps(
            {
                "trace_path": summary.trace_path,
                "episodes": summary.episodes,
                "scenarios": summary.scenarios,
                "rows": summary.rows,
                "backend": cfg.backend.backend,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
