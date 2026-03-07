from __future__ import annotations

from typing import Dict

import numpy as np

from ..trace_schema import read_trace_jsonl


def summarize_trajectories(trace_jsonl: str) -> Dict[str, float]:
    rows = read_trace_jsonl(trace_jsonl)
    if not rows:
        return {"rows": 0.0}
    moral = np.asarray([float(r.get("moral_score_proxy", 0.0)) for r in rows], dtype=np.float32)
    consistency = np.asarray([float(r.get("consistency_score_proxy", 0.0)) for r in rows], dtype=np.float32)
    useful = np.asarray([float(r.get("usefulness_score_proxy", 0.0)) for r in rows], dtype=np.float32)
    return {
        "rows": float(len(rows)),
        "moral_mean": float(moral.mean()),
        "moral_p10": float(np.quantile(moral, 0.1)),
        "consistency_mean": float(consistency.mean()),
        "usefulness_mean": float(useful.mean()),
    }
