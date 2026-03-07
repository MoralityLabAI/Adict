from __future__ import annotations

from typing import Dict

from ..trace_schema import read_trace_jsonl
from .moral_reasoning import moral_reasoning_rubric


def evaluate_storyworld_trace(trace_jsonl: str) -> Dict[str, float]:
    rows = read_trace_jsonl(trace_jsonl)
    rubric = moral_reasoning_rubric(rows)
    scenarios = len({str(r.get("scenario_id", "")) for r in rows})
    episodes = len({str(r.get("episode_id", "")) for r in rows})
    rubric["scenarios"] = float(scenarios)
    rubric["episodes"] = float(episodes)
    return rubric
