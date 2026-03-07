from __future__ import annotations

from typing import Dict, List

import numpy as np

from .utils import utc_ts


def make_synthetic_trace_rows(n_episodes: int = 8, steps: int = 12, seed: int = 7) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for ep in range(n_episodes):
        moral = 0.45 + 0.1 * rng.random()
        consistency = 0.4 + 0.1 * rng.random()
        usefulness = 0.5 + 0.1 * rng.random()
        for t in range(steps):
            drift = float(rng.normal(0, 0.03))
            moral = float(np.clip(moral + drift, 0, 1))
            consistency = float(np.clip(consistency + 0.5 * drift, 0, 1))
            usefulness = float(np.clip(usefulness - 0.3 * abs(drift), 0, 1))
            rows.append(
                {
                    "ts": utc_ts(),
                    "episode_id": f"synthetic-ep-{ep:04d}",
                    "step_idx": t,
                    "scenario_id": "synthetic_dilemma",
                    "prompt_summary": "synthetic prompt",
                    "model_output_summary": "synthetic output",
                    "action_summary": "action",
                    "moral_score_proxy": moral,
                    "usefulness_score_proxy": usefulness,
                    "consistency_score_proxy": consistency,
                    "manipulation_risk_proxy": 1.0 - consistency,
                    "story_goal_progress_proxy": usefulness,
                    "hidden_state_summary": rng.normal(size=(32,)).astype(float).tolist(),
                    "bounded_history_summary": "synthetic history",
                    "intervention_needed": bool(moral < 0.35 or consistency < 0.3),
                    "backend": "synthetic",
                    "oracle": "none",
                    "latency_ms": float(20 + 5 * rng.random()),
                }
            )
    return rows
