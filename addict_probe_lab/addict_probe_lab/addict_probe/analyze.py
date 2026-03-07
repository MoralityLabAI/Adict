from __future__ import annotations

from typing import Dict

import numpy as np

from .trace_schema import DEFAULT_LAYOUT, jsonl_to_btf


def summarize_trace_file(path: str) -> Dict[str, float]:
    btf, episode_ids = jsonl_to_btf(path, layout=DEFAULT_LAYOUT)
    if btf.size == 0:
        return {"episodes": 0.0, "steps": 0.0, "feature_dim": float(DEFAULT_LAYOUT.total_dim)}
    return {
        "episodes": float(len(episode_ids)),
        "steps": float(btf.shape[1]),
        "feature_dim": float(btf.shape[2]),
        "mean_abs_feature": float(np.mean(np.abs(btf))),
    }
