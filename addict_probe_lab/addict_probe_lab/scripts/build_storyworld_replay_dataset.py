#!/usr/bin/env python
"""Build a bounded JSONL replay dataset from storyworld traces for short adapter training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.trace_schema import read_trace_jsonl
from addict_probe.harness.response_contract import parse_contract_output, strip_think_blocks


def _bucket(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:10], 16) / float(16**10)


def _composite_score(row: Dict[str, Any]) -> float:
    return (
        0.45 * float(row.get("moral_score_proxy", 0.0))
        + 0.35 * float(row.get("consistency_score_proxy", 0.0))
        + 0.20 * float(row.get("usefulness_score_proxy", 0.0))
        - 0.15 * float(row.get("manipulation_risk_proxy", 0.0))
        + 0.10 * float(row.get("guardrail_score", 0.0))
    )


def _messages_for_row(row: Dict[str, Any], transition: Dict[str, Any]) -> List[Dict[str, str]]:
    guardrail_atoms = ", ".join(str(v) for v in transition.get("guardrail_atoms", [])[:6]) or "none"
    suppress_atoms = ", ".join(str(v) for v in transition.get("suppress_atoms", [])[:6]) or "none"
    system = (
        "You are training a short-context constitutional storyworld adapter.\n"
        "Priorities:\n"
        "- protect affected parties without manipulative shortcuts\n"
        "- keep action aligned with stated principle across turns\n"
        "- stay concrete, useful, and non-preachy\n"
        f"- reinforce successful lens motifs: {guardrail_atoms}\n"
        f"- avoid drift motifs: {suppress_atoms}\n"
        "Respond exactly in this format:\n"
        "INTENDED_ACTION: ...\n"
        "DIALOGUE: ...\n"
        "MORAL_LOGIC: ..."
    )
    user = (
        f"Scenario: {row.get('scenario_id', '')}\n"
        f"Prompt summary:\n{row.get('prompt_summary', '')}\n\n"
        f"Bounded history:\n{row.get('bounded_history_summary', '')}\n\n"
        "Produce the next action, dialogue, and moral logic."
    ).strip()
    raw_assistant = str(row.get("model_output_summary") or row.get("action_summary") or "").strip()
    parsed = parse_contract_output(raw_assistant)
    assistant = str(parsed.get("normalized_text", strip_think_blocks(raw_assistant))).strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def build_replay_rows(
    trace_path: str | Path,
    transition_report_path: str | Path,
    min_rows: int = 24,
) -> List[Dict[str, Any]]:
    rows = read_trace_jsonl(trace_path)
    transition = json.loads(Path(transition_report_path).read_text(encoding="utf-8-sig"))
    if not rows:
        return []
    rows = [r for r in rows if bool(r.get("effective_valid_action", r.get("valid_action", True)))]
    if not rows:
        return []
    scored = sorted(rows, key=_composite_score, reverse=True)
    cutoff = max(min_rows, len(scored) // 2)
    selected = scored[:cutoff] if len(scored) > cutoff else scored
    out: List[Dict[str, Any]] = []
    for row in selected:
        row_id = f"{row.get('episode_id', '')}:{row.get('step_idx', 0)}"
        out.append(
            {
                "id": row_id,
                "scenario_id": row.get("scenario_id", ""),
                "episode_id": row.get("episode_id", ""),
                "step_idx": int(row.get("step_idx", 0)),
                "composite_score": _composite_score(row),
                "guardrail_score": float(row.get("guardrail_score", 0.0)),
                "messages": _messages_for_row(row, transition),
            }
        )
    return out


def write_splits(rows: List[Dict[str, Any]], out_dir: str | Path) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "valid": [], "test": []}
    for row in rows:
        bucket = _bucket(str(row["id"]))
        if bucket < 0.84:
            splits["train"].append(row)
        elif bucket < 0.92:
            splits["valid"].append(row)
        else:
            splits["test"].append(row)

    if not splits["valid"] and splits["train"]:
        splits["valid"].append(splits["train"].pop())
    if not splits["test"] and len(splits["train"]) > 1:
        splits["test"].append(splits["train"].pop())

    for name, items in splits.items():
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    manifest = {
        "dataset_dir": str(out),
        "counts": {name: len(items) for name, items in splits.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build replay JSONL dataset from storyworld traces")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--transition-report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-rows", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_replay_rows(args.trace, args.transition_report, min_rows=args.min_rows)
    manifest = write_splits(rows, args.out_dir)
    manifest["rows_selected"] = len(rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
